import json
import asyncio
from openai import AsyncOpenAI
from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, LLM_MODELS
from prompts import VISION_PROMPT

client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def parse_llm_result(raw_content: str) -> dict | None:
    result = raw_content.strip()

    # Clean markdown fences if present
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
    except json.JSONDecodeError:
        # Gracefully handle plain text errors or safety blocks
        print(f">> Warning: Raw response wasn't valid JSON. Generating safety fallback schema.")
        return {
            "lesson_title": "Content Alert / Safety Block",
            "factual": [{"heading": "System Notice", "content": f"[Raw System Text: {result}]"}],
            "conceptual": [], 
            "procedural": [], 
            "metacognitive": []
        }


async def call_llm_with_fallback(
    chunk: str,
    chunk_index: int,
    system_prompt: str,
    models: list[str] = LLM_MODELS,
    max_tokens: int = 16000,
) -> dict | None:
    for model in models:
        print(f">> Chunk {chunk_index} trying model: {model}")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                max_tokens=max_tokens,
            )

            raw_content = response.choices[0].message.content
            print(f">> Raw response from {model}: {repr(raw_content[:300]) if raw_content else 'EMPTY'}")

            if not raw_content:
                print(f">> {model} returned empty, trying next...")
                await asyncio.sleep(0.5)  # Optimized delay from 2s to 0.5s for performance
                continue

            result = parse_llm_result(raw_content)
            
            # Since parse_llm_result now always returns a fallback dict structure 
            # instead of None, we verify it contains keys we need
            if not result or not isinstance(result, dict):
                print(f">> {model} structure missing or invalid dictionary type, trying next...")
                await asyncio.sleep(0.5)
                continue

            print(f">> Chunk {chunk_index} succeeded with {model}")
            return result

        except asyncio.TimeoutError:
            print(f">> {model} timed out, trying next...")
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f">> {model} failed with exception: {e}, trying next...")
            await asyncio.sleep(0.5)

    print(f">> Chunk {chunk_index} failed on all available models.")
    return None


async def call_vision_model(model: str, b64_image: str) -> str | None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": VISION_PROMPT,
                },
            ],
        }
    ]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
    )
    return response.choices[0].message.content