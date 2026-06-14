

from graph.state import EductState
from services.pdf_utils import build_full_context


def context_node(state: EductState):
    full_context = build_full_context(state['extracted_pages'], state['image_descriptions'])
    return {"full_context": full_context}