import asyncio
import base64
import fitz
import pymupdf4llm
from config import VISION_MODELS
from services.llm_client import call_vision_model


async def extract_markdown_pages(file_path: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    print(f">> Extracting text from: {file_path}")
    result = await loop.run_in_executor(
        None,
        lambda: pymupdf4llm.to_markdown(doc=file_path, page_chunks=True),
    )
    print(f">> Text extraction complete. Pages: {len(result)}")
    return result


def render_page_as_base64(pdf_path: str, page_num: int) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def get_image_page_numbers(pdf_path: str) -> list[int]:
    with fitz.open(pdf_path) as doc:
        return [
            page_num
            for page_num, page in enumerate(doc)
            if page.get_images(full=True)
        ]


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


def build_full_context(
    pages: list[dict],
    image_descriptions: dict[int, str | None],
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


async def one_page_process(pdf_path: str, page_num: int) -> tuple[int, str | None]:
    loop = asyncio.get_event_loop()
    b64_image = await loop.run_in_executor(None, render_page_as_base64, pdf_path, page_num)
    description = None
    for model in VISION_MODELS:
        try:
            print(f">> Trying vision model: {model}")
            raw = await call_vision_model(model, b64_image)
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


async def describe_page_images(pdf_path: str) -> dict[int, str | None]:
    try:
        with fitz.open(pdf_path) as doc:
            img_pages = [page_num for page_num, page in enumerate(doc) if page.get_images(full=True)]
        
        tasks = [one_page_process(pdf_path, page_num) for page_num in img_pages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Clean parsing:
        valid_descriptions = {}
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            page_num, desc = result  # Clean tuple unpacking
            valid_descriptions[page_num] = desc
            
        return valid_descriptions

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f">> Image description failed: {e}")
        return {}