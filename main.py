import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from graph.builder import build_graph
from graph.state import EductState
from Database.educt_db import get_db, init_db
from Database.models import markdownFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables safely on app startup
    init_db()
    yield

app = FastAPI(title="EDUCTAR API", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api", tags=["EDUCTAR API"])

# Compile the graph
graph = build_graph()


@router.get("/")
async def root():
    return {"message": "Welcome to Eductar API. Visit /docs for documentation"}


def convert_classified_to_markdown(classified: dict) -> str:
    title = classified.get("lesson_title", "Untitled Lecture")
    md = f"# {title}\n\n"

    for category in ["factual", "conceptual", "procedural", "metacognitive"]:
        md += f"## {category.upper()}\n\n"
        blocks = classified.get(category, [])
        if not blocks:
            md += "*No content in this category.*\n\n"
        for block in blocks:
            heading = block.get("heading", "")
            content = block.get("content", "")
            if heading:
                md += f"### {heading}\n"
            md += f"{content}\n\n"
    return md


@router.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)

    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        # Save file to temp_files/
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize workflow state
        initial_state = EductState(
            file_path=temp_file_path,
            extracted_pages=[],
            image_descriptions={},
            full_context="",
            chunks=[],
            classified={},
            revision_count=0,
            approved=False,
        )

        # Execute workflow
        result = await graph.ainvoke(initial_state)

        # Persist approved classifications to the database
        classified_data = result.get("classified", {})
        if classified_data and classified_data.get("lesson_title"):
            markdown_content = convert_classified_to_markdown(classified_data)

            with get_db() as db:
                db_file = (
                    db.query(markdownFiles)
                    .filter(markdownFiles.file_path == temp_file_path)
                    .first()
                )
                if db_file:
                    db_file.filename = file.filename
                    db_file.content = markdown_content
                else:
                    db_file = markdownFiles(
                        file_path=temp_file_path,
                        filename=file.filename,
                        content=markdown_content,
                    )
                    db.add(db_file)

        return {
            "status": "success",
            "approved": result.get("approved", False),
            "revision_count": result.get("revision_count", 0),
            "classified": classified_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

