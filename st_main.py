from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from documents import documents
import streamlit as st


encoder = SentenceTransformer("all-MiniLM-L6-v2")

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


def search_document(searchterm: str, year_filter: int, num_results: int = 3):
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=encoder.encode(f"{searchterm}"),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="year", range=models.Range(gte=year_filter))
            ]
        ),
        limit=num_results,
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


st.header("QS_search")
st.subheader("Semantic Search With Qdrant and Streamlit")
st.write("Search in a list of  science fiction books")
search_input = st.text_input("Search")
year_filter = st.number_input("Filter by Year (Leave blank to ignore)", min_value=None)
num_results = st.number_input("Number of Results", min_value=1)
if search_input:
    st.markdown(
        "<h1 style='color: purple;'>Search Results:</h1>", unsafe_allow_html=True
    )
    for result in search_document(search_input, year_filter, num_results):

        st.markdown(f"**Name:** {result['name']}", unsafe_allow_html=True)
        st.markdown(f"**Description:** {result['description']}", unsafe_allow_html=True)
        st.markdown(f"**Author:** {result['author']}", unsafe_allow_html=True)
        st.markdown(f"**Score:** {result['score']:.2f}", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid purple;'>", unsafe_allow_html=True)
