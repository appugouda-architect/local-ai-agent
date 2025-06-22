from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("kids_cycle_reviews.csv")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vecor_store = Chroma(
    collection_name="cycle_shop_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vecor_store.add_documents(documents=documents, ids=ids)

retriever = vecor_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
