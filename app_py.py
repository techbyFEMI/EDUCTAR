from __future__ import annotations
import fitz
import json
import os
import base64
import time
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.dialects.postgresql import insert
import pymupdf4llm
from openai import OpenAI

from Database import educt_db
from Database.educt_db import Base, sessionLocal
from Database.models import markdownFiles

load_dotenv()

LLM_MODELS = [
   
    "arcee-ai/trinity-large-preview:free",
    "openrouter/hunter-alpha",
    "mistralai/mistral-small-3.1-24b-instruct:free",
     "nvidia/nemotron-nano-9b-v2:free",
]

VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
     "google/gemma-3-4b-it:free",
   
]

VISION_PROMPT = """
You are analyzing a page from an educational lecture PDF.
Describe what you see in this image in detail — any diagrams, charts, 
figures, tables, or visual elements. Focus on the educational content 
they represent. Be specific about labels, relationships shown, and 
what concept the visual is explaining.
If there are no meaningful visuals (just text), respond with "No significant visual content."
Keep your description under 200 words.
"""

PROMPT = """
You are an expert educational content organizer specializing in Bloom's Taxonomy.

You will receive the full content of a lecture note including text and image descriptions.
Your job is to:
1. Deeply understand the full lesson being taught including what the images show
2. Rewrite the EXACT same content reorganized into Bloom's Taxonomy order
3. Use the EXACT same words and explanations from the original text
4. For image descriptions, include them as [IMAGE: description] in the relevant section
5. Reorganize the existing content into this learning progression:

FACTUAL — Basic facts, definitions, terminology, specific details
CONCEPTUAL — Theories, principles, relationships, classifications, diagrams explaining concepts
PROCEDURAL — Steps, processes, methods, sequences, how things work
METACOGNITIVE — Reflection, overviews, self-awareness, learning strategies

Rules:
- Every piece of content must appear in the output
- Use exact original wording for text — no summarizing, no paraphrasing
- Place image descriptions in the most relevant Bloom category
- Maintain logical flow within each category

Return ONLY this JSON, no extra text, no markdown fences:
{
    "lesson_title": "title of the lecture",
    "factual": [
        {"heading": "section heading if any", "content": "exact original text or [IMAGE: description]"}
    ],
    "conceptual": [
        {"heading": "section heading if any", "content": "exact original text or [IMAGE: description]"}
    ],
    "procedural": [
        {"heading": "section heading if any", "content": "exact original text or [IMAGE: description]"}
    ],
    "metacognitive": [
        {"heading": "section heading if any", "content": "exact original text or [IMAGE: description]"}
    ]
}
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=180.0
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=educt_db.engine)


# ── Helpers ──────────────────────────────────────────────────

def markdown_extractor(file_path: str):
    print(f">> Extracting text from: {file_path}")
    result = pymupdf4llm.to_markdown(doc=file_path, page_chunks=True)
    print(f">> Text extraction complete. Pages: {len(result)}")
    return result


def render_page_as_base64(pdf_path: str, page_num: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def describe_page_images(pdf_path: str) -> dict[int, str | None]:
    doc = fitz.open(pdf_path)
    page_descriptions: dict[int, str | None] = {}
    for page_num,page in enumerate(doc):
        imgcheck=page.get_images(full=True)
        if not imgcheck:
            continue

        
        print(f">> Vision model analyzing page {page_num + 1}...")
        b64_image = render_page_as_base64(pdf_path, page_num)
        description = None

        for model in VISION_MODELS:
                    try:
                        print(f">> Trying vision model: {model}")
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{b64_image}"
                                            }
                                        },
                                        {
                                            "type": "text",
                                            "text": VISION_PROMPT
                                        }
                                    ]
                                }
                            ],
                            max_tokens=300,
                        )

                        raw = response.choices[0].message.content
                        if not raw:
                            print(f">> {model} returned empty, trying next...")
                            time.sleep(2)
                            continue

                        result = raw.strip()
                        if "No significant visual content" in result:
                            description = None
                        else:
                            print(f">> Page {page_num + 1} described by {model}")
                            description = result

                        break

                    except Exception as e:
                        print(f">> {model} failed: {e}, trying next...")
                        time.sleep(2)
                        continue

        page_descriptions[page_num + 1] = description
        time.sleep(1)

    doc.close()
    return page_descriptions


def build_full_context(
    pages: list[dict],
    image_descriptions: dict[int, str | None]
) -> str:
    full_context = ""

    for i, page_data in enumerate(pages):
        page_num = page_data.get("metadata", {}).get("page", i + 1)
        text = page_data.get("text", "").strip()
        img_desc = image_descriptions.get(page_num)

        if text:
            full_context += text + "\n\n"
        if img_desc:
            full_context += f"[IMAGE: {img_desc}]\n\n"

    return full_context


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    chunks = []
    current = ""

    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
            current = paragraph
        else:
            current += "\n\n" + paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def parse_llm_result(raw_content: str) -> dict | None:
    result = raw_content.strip()

    if "```" in result:
        parts = result.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part.startswith("{"):
                result = part
                break

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f">> JSON parse error: {e} | raw: {result[:300]}")
        return None


def call_llm_with_fallback(chunk: str, chunk_index: int) -> dict | None:
    for model in LLM_MODELS:
        print(f">> Chunk {chunk_index} trying model: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=16000,
            )

            raw_content = response.choices[0].message.content
            print(f">> Raw response from {model}: {repr(raw_content[:300]) if raw_content else 'EMPTY'}")

            if not raw_content:
                print(f">> {model} returned empty, trying next...")
                time.sleep(2)
                continue

            result = parse_llm_result(raw_content)
            if not result:
                print(f">> {model} returned invalid JSON, trying next...")
                time.sleep(2)
                continue

            print(f">> Chunk {chunk_index} succeeded with {model}")
            return result

        except Exception as e:
            print(f">> {model} failed: {e}, trying next...")
            time.sleep(2)
            continue

    print(f">> Chunk {chunk_index} failed on all models")
    return None


def build_txt_output(classified: dict) -> str:
    bloom_labels = {
        "factual": "FACTUAL KNOWLEDGE",
        "conceptual": "CONCEPTUAL KNOWLEDGE",
        "procedural": "PROCEDURAL KNOWLEDGE",
        "metacognitive": "METACOGNITIVE KNOWLEDGE"
    }

    output = ""
    output += f"{classified.get('lesson_title', 'Lecture Notes')}\n"
    output += "=" * 60 + "\n\n"

    for category in ["factual", "conceptual", "procedural", "metacognitive"]:
        blocks = classified.get(category, [])
        if not blocks:
            continue

        output += f"{bloom_labels[category]}\n"
        output += "-" * 40 + "\n\n"

        for block in blocks:
            heading = block.get("heading", "")
            content = block.get("content", "")

            if heading:
                output += f"{heading}\n"
            if content:
                output += f"{content}\n"
            output += "\n"

        output += "\n"

    return output


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Welcome to Eductar API. Visit /docs for documentation"}


# @app.get("/test")
# async def frontend():
#     return FileResponse("test.html")


@app.post("/upload_and_extract")
async def upload_and_extract(file: UploadFile = File(...)):
    os.makedirs("temp_files", exist_ok=True)

    file_path = f"temp_files/{file.filename}"
    with open(file_path, "wb") as buffer:
        contents = await file.read()
        buffer.write(contents)

    # 1. Extract text
    extracted_content = markdown_extractor(file_path)
    full_markdown = "\n\n".join([page["text"] for page in extracted_content])

    # 2. Save to DB
    db = sessionLocal()
    try:
        stmt = insert(markdownFiles).values(
            file_path=file_path,
            filename=file.filename,
            content=full_markdown
        ).on_conflict_do_update(
            index_elements=['file_path'],
            set_=dict(content=full_markdown)
        )
        db.execute(stmt)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

    # 3. Vision model analyzes every page
    print(">> Starting image analysis...")
    image_descriptions = describe_page_images(file_path)

    # 4. Build combined context
    full_context = build_full_context(extracted_content, image_descriptions)
    print(f">> Full context length: {len(full_context)} characters")

    # 5. Chunk and send to LLM
    chunks = chunk_text(full_context, max_chars=400)
    print(f">> Split into {len(chunks)} chunks")

    all_classified: dict = {
        "lesson_title": file.filename,
        "factual": [],
        "conceptual": [],
        "procedural": [],
        "metacognitive": []
    }

    for i, chunk in enumerate(chunks):
        print(f">> Processing chunk {i + 1}/{len(chunks)}...")

        chunk_classified = call_llm_with_fallback(chunk, i + 1)

        if not chunk_classified:
            print(f">> Chunk {i + 1} failed on all models, skipping...")
            continue

        if i == 0:
            all_classified["lesson_title"] = chunk_classified.get(
                "lesson_title", file.filename
            )

        for category in ["factual", "conceptual", "procedural", "metacognitive"]:
            all_classified[category].extend(chunk_classified.get(category, []))

        time.sleep(4)

    if not any(all_classified[cat] for cat in ["factual", "conceptual", "procedural", "metacognitive"]):
        return {"error": "All chunks failed on all models. Check OpenRouter credits or model availability."}

    # 6. Build and save txt output
    txt_output = build_txt_output(all_classified)
    output_filename = file.filename.replace(".pdf", "") + "_bloom.txt"
    output_path = f"temp_files/{output_filename}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt_output)

    print(f">> Output TXT saved: {output_path}")

    return FileResponse(
        path=output_path,
        media_type="text/plain",
        filename=output_filename
    )