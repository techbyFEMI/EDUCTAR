import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json
import fitz
import base64
import pymupdf4llm
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODELS=[
    "arcee-ai/trinity-large-thinking:free",
    "nvidia/nemotron-nano-9b-v2:free"
]

VISION_MODELS=[
    "arcee-ai/vision-1:free",
    "google/gemma-3-4b-it:free"
]


client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY") 
)

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


class EductState(TypedDict):
    file_path:str
    extracted_pages:list[dict]
    image_descriptions:dict
    full_context:str
    chunks:list[str]
    classified:dict
    revision_count:int
    approved:bool

type pagedesc = dict[int, str | None]

async def markdown_extractor(file_path: str):
    loop = asyncio.get_event_loop()
    print(f">> Extracting text from: {file_path}")
    result = await loop.run_in_executor(
        None, 
        lambda: pymupdf4llm.to_markdown(doc=file_path, page_chunks=True)
    )
    print(f">> Text extraction complete. Pages: {len(result)}")
    return result

async def extract_node(state:EductState):
    result = await markdown_extractor(state['file_path'])
    return {"extracted_pages": result}

def render_page_as_base64(pdf_path: str, page_num: int) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        mat =fitz.Matrix(2, 2)
        pix =page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

    return base64.b64encode(img_bytes).decode('utf-8')

async def call_vision_model(model:str, b64_image:str)->str|None:
    messages=[
            {
                "role": "user",
                "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                            }
                                },
                                {
                                    "type": "text",
                                    "text": VISION_PROMPT,
                                },
                            ],
            }
    ]
    response =await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
                
                    )
      
    return response.choices[0].message.content

async def one_page_process(pdf_path:str,page_num:int)->tuple[int, str | None]:
        loop =asyncio.get_event_loop()
        b64_image = await loop.run_in_executor(None, render_page_as_base64, pdf_path, page_num)
        description = None
        for model in VISION_MODELS:
                    try:
                        print(f">> Trying vision model: {model}")
                      
  
                        raw = await call_vision_model(model,b64_image)
                        if not raw:
                            print(f">> {model} returned empty, trying next...")
                            await asyncio.sleep(2)
                            continue

                        result = raw.strip()
                        if "No significant visual content" in result:
                            description = None
                        else:
                            print(f">> Page {page_num + 1} described by {model}")
                            description = result
                            break
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        print(f">> {model} timed out: trying next...")
                        await asyncio.sleep(2)
                        continue
                    except Exception as e:
                        print(f">> {model} failed: {e}, trying next...")
                        await asyncio.sleep(2)
                        continue
        return (page_num + 1, description)

async def describe_page_images(pdf_path: str, ) -> pagedesc:
    
    try:
        with fitz.open(pdf_path) as doc:
            img_pages=[]
            for page_num,page in enumerate(doc):
                if page.get_images(full=True):
                    img_pages.append((page_num))
        task=[one_page_process(pdf_path,page_num) for page_num in img_pages]
        results=await asyncio.gather(*task,return_exceptions=True)
        return {page_num: desc for result in results if not isinstance(result, Exception)
                for page_num, desc in [result]}

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f">> Image description failed: {e}")
        return {}
async def vision_node(state:EductState):
    descriptions= await describe_page_images(state['file_path'])
    return {"image_descriptions":descriptions}

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
def context_node(state:EductState):
    full_context= build_full_context(state['extracted_pages'], state['image_descriptions'])
    return {"full_context": full_context}


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
def chunking_node(state:EductState):
    chunks = chunk_text(state['full_context'])
    return {"chunks": chunks}


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


async def call_llm_with_fallback(chunk: str, chunk_index: int) -> dict | None:
    for model in LLM_MODELS:
        print(f">> Chunk {chunk_index} trying model: {model}")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=16000,
            )

            raw_content =  response.choices[0].message.content
            print(f">> Raw response from {model}: {repr(raw_content[:300]) if raw_content else 'EMPTY'}")

            if not raw_content:
                print(f">> {model} returned empty, trying next...")
                await asyncio.sleep(2)
                continue

            result = parse_llm_result(raw_content)
            if not result:
                print(f">> {model} returned invalid JSON, trying next...")
                await asyncio.sleep(2)
                continue

            print(f">> Chunk {chunk_index} succeeded with {model}")
            return result

        except asyncio.TimeoutError:
            print(f">> {model} timed out: trying next...")
            await asyncio.sleep(2)
            continue
        except asyncio.CancelledError:
            print(f">> {model} call cancelled.")
            raise
        except Exception as e:
            print(f">> {model} failed: {e}, trying next...")
            await asyncio.sleep(2)
            continue

    print(f">> Chunk {chunk_index} failed on all models")
    return None


async def classify_node(state: EductState):
    chunks = state['chunks']
    tasks = [call_llm_with_fallback(chunk, idx) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_classified = {
        "lesson_title": "",
        "factual": [], "conceptual": [],
        "procedural": [], "metacognitive": []
    }
    seen = set()

    for idx, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            continue
        if idx == 0:
            all_classified["lesson_title"] = result.get("lesson_title", "")
        for category in ["factual", "conceptual", "procedural", "metacognitive"]:
            for block in result.get(category, []):
                key = block.get("content", "")[:120].strip()
                if key and key not in seen:
                    seen.add(key)
                    all_classified[category].append(block)

    return {"classified": all_classified}  





GRADE_PROMPT = """
You are an expert educational content grader specializing in Bloom's Taxonomy.

Review the classified content below and evaluate it.

Return ONLY this JSON, no extra text:
{"score": <int 1-10>, "feedback": "<brief explanation>"}

Score criteria:
- 9-10: All four categories populated, content correctly placed
- 7-8: Most categories correct, minor misplacements  
- 5-6: Some categories empty or content misplaced
- Below 5: Major classification errors
"""
async def grade_node(state: EductState):
    classified = state["classified"]
    response = await client.chat.completions.create(
        model=LLM_MODELS[0],
        messages=[
            {"role": "system", "content": GRADE_PROMPT},
            {"role": "user", "content": json.dumps(classified)}
        ],
        max_tokens=200,
    )
    raw = response.choices[0].message.content
    parsed = parse_llm_result(raw)
    score = parsed.get("score", 0) if parsed else 0
    return {
        "approved": score >= 7,
        "revision_count": state["revision_count"] + 1
    }



def should_continue(state: EductState) -> str:
    if state["approved"] or state["revision_count"] >= 2:
        return END
    return "classify"


                                                
builder = StateGraph(EductState)

builder.add_node("extract", extract_node)
builder.add_node("vision", vision_node)
builder.add_node("context", context_node)
builder.add_node("chunk", chunking_node)
builder.add_node("classify", classify_node)
builder.add_node("grade", grade_node)

builder.set_entry_point("extract")

builder.add_edge("extract", "vision")
builder.add_edge("vision", "context")
builder.add_edge("context", "chunk")
builder.add_edge("chunk", "classify")
builder.add_edge("classify", "grade")
builder.add_conditional_edges("grade", should_continue)

graph = builder.compile()