from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import streamlit as st
import json

with open("resumes.json", "r") as json_file:
    documents = json.load(json_file)


encoder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(":memory:")



# create a collection
qdrant.recreate_collection(
    collection_name="my_resumes",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE
    ),
)

# upload data to the collection
qdrant.upload_records(
    collection_name="my_resumes",
    records=[
        models.Record(
            id=idx, vector=encoder.encode(doc["Resume"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

@st.cache_resource
def search_document(searchterm: str, num_results: int = 10):
    hits = qdrant.search(
        collection_name="my_resumes",
        query_vector=encoder.encode(f"{searchterm}"),
        limit=num_results,
    )
    result = [
        {
            "ID": hit.payload["ID"],
            "Resume": hit.payload["Resume"],
            "Category": hit.payload["Category"],
            "score": hit.score,
        }
        for hit in hits
    ]
    return result if searchterm else []


st.header("AI Powered Search Engine")
st.subheader("RepoTech Resume search")
st.write(
    "Search within around 2500 resumes, search by skills,project or any field you can think of "
)
search_input = st.text_input("Search")
unique_name_of_categories = ('HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
       'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE',
       'BPO', 'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE',
       'CHEF', 'FINANCE', 'APPAREL', 'ENGINEERING', 'ACCOUNTANT',
       'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING', 'ARTS', 'AVIATION')
num_results = st.number_input(
    "Number of returned Resumes", min_value=1, max_value=30, value=5
)

if search_input : 
    st.markdown(
        "<h1 style='color: purple;'>Search Results:</h1>", unsafe_allow_html=True
    )
    for result in search_document(search_input, num_results):
        print(result)
        st.markdown(f"**ID:** {result['ID']}", unsafe_allow_html=True)
        st.markdown(
            f"**Resume Content:** {result['Resume']}", unsafe_allow_html=True
        )
        st.markdown(f"**Category:** {result['Category']}", unsafe_allow_html=True)
