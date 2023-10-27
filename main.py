from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from qdrant_client import models, QdrantClient
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from documents import documents

# Initizlie FastAPI
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchBar(BaseModel):
    searchterm: str


# Initizlie Jinja2Templates
templates = Jinja2Templates(directory="templates")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


# def storage location
qdrant = QdrantClient(":memory:")


# create a collection
qdrant.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE
    ),
)
# upload data to the collection
qdrant.upload_records(
    collection_name="my_books",
    records=[
        models.Record(
            id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)


def search_document(searchterm: str):
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=encoder.encode(f"{searchterm}"),
        limit=3,
    )
    result = [
        {
            "name": hit.payload["name"],
            "description": hit.payload["description"],
            "author": hit.payload["author"],
            "score": hit.score,
        }
        for hit in hits
    ]
    return result if searchterm else []


@app.get("/")
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.post("/search/{searchterm}", response_class=HTMLResponse)
async def search_results(request: Request, searchterm: str):
    results = search_document(searchterm)
    return templates.TemplateResponse(
        "results.html", {"request": request, "results": results}
    )
