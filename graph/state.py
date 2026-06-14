from typing import TypedDict


class EductState(TypedDict):
    file_path: str
    extracted_pages: list[dict]
    image_descriptions: dict
    full_context: str
    chunks: list[str]
    classified: dict
    revision_count: int
    approved: bool