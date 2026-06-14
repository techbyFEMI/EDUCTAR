from graph.state import EductState
from services.pdf_utils import extract_markdown_pages as markdown_extractor

async def extract_node(state: EductState):
    result = await markdown_extractor(state['file_path'])
    return {"extracted_pages": result}