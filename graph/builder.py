from langgraph.graph import StateGraph, END
from graph.state import EductState
from nodes.extract import extract_node
from nodes.vision import vision_node
from nodes.context import context_node
from nodes.chunk import chunking_node as chunk_node
from nodes.classify import classify_node
from nodes.grade import grade_node


def should_continue(state: EductState) -> str:
    if state["approved"] or state["revision_count"] >= 2:
        return END
    return "classify"


def build_graph() -> StateGraph:
    builder = StateGraph(EductState)

    builder.add_node("extract", extract_node)
    builder.add_node("vision", vision_node)
    builder.add_node("context", context_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("classify", classify_node)
    builder.add_node("grade", grade_node)

    builder.set_entry_point("extract")

    builder.add_edge("extract", "vision")
    builder.add_edge("vision", "context")
    builder.add_edge("context", "chunk")
    builder.add_edge("chunk", "classify")
    builder.add_edge("classify", "grade")
    builder.add_conditional_edges("grade", should_continue)

    return builder.compile()