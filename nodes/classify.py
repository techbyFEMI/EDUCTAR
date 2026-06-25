from services.llm_client import call_llm_with_fallback
from graph.state import EductState
from prompts import CLASSIFIER_PROMPT
import asyncio

BATCH_SIZE = 5
BATCH_DELAY = 4  # seconds between batches to respect rate limits

async def classify_node(state: EductState):
    chunks = state['chunks']
    
    #  Pass CLASSIFIER_PROMPT as the required third positional argument
    tasks = [call_llm_with_fallback(chunk, idx, CLASSIFIER_PROMPT) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_classified = {
        "lesson_title": "Classified Lecture Notes",
        "factual": [], "conceptual": [],
        "procedural": [], "metacognitive": []
    }
    seen = set()
    title_set = False

    for idx, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            print(f">> Warning: Chunk {idx} failed completely. Skipping.")
            continue
            
        if not isinstance(result, dict):
            print(f">> Warning: Chunk {idx} did not return a dictionary. Skipping.")
            continue

        if not title_set and result.get("lesson_title"):
            all_classified["lesson_title"] = result.get("lesson_title", "").strip()
            title_set = True

        for category in ["factual", "conceptual", "procedural", "metacognitive"]:
            for block in result.get(category, []):
                if isinstance(block, dict):
                    key = block.get("content", "")[:120].strip()
                    if key and key not in seen:
                        seen.add(key)
                        all_classified[category].append(block)

    return {"classified": all_classified}