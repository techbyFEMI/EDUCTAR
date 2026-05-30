from typing import TypedDict
from langgraph.graph import StateGraph, END


class EductState(TypedDict):
    file_path:str
    extracted_pages:list[dict]
    image_descriptions:dict
    full_context:str
    chunks:list[str]
    classified:dict
    revision_count:int
    approved:bool

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

    return base64.b64encode(img_bytes)


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

async def describe_page_images(pdf_path: str) -> pagedesc:
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

