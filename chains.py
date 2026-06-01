
import asyncio



PROMPTGRADE_PROMPT = """
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

grade_prompt=
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
