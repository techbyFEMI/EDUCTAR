from services.llm_client import call_llm_with_fallback
from graph.state import EductState
from prompts import CLASSIFIER_PROMPT
import asyncio

BATCH_SIZE = 5
BATCH_DELAY = 4  # seconds between batches to respect rate limits

async def classify_node(state: EductState):
    chunks = state['chunks']

    # Process in batches to respect free-tier rate limits
    results = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_tasks = [
            call_llm_with_fallback(chunk, i + j, CLASSIFIER_PROMPT)
            for j, chunk in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
        if i + BATCH_SIZE < len(chunks):
            print(f">> Batch {i // BATCH_SIZE + 1} done. Waiting {BATCH_DELAY}s before next batch...")
            await asyncio.sleep(BATCH_DELAY)

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