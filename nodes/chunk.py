from services.pdf_utils import chunk_text
from graph.state import EductState


def chunking_node(state: EductState):
    chunks = chunk_text(state['full_context'])
    return {"chunks": chunks}