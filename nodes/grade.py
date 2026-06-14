import json
from graph.state import EductState
from services.llm_client import client, parse_llm_result
from config import LLM_MODELS
from prompts import GRADER_PROMPT

async def grade_node(state: EductState):
    classified = state["classified"]
    response = await client.chat.completions.create(
        model=LLM_MODELS[0],
        messages=[
            {"role": "system", "content": GRADER_PROMPT},
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
