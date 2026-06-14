from graph.state import EductState
from services.pdf_utils import describe_page_images

async def vision_node(state: EductState):
    descriptions = await describe_page_images(state['file_path'])
    return {"image_descriptions": descriptions}