import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("🤖 Nick'x AI ChatBot")

# Load model only once
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# Load knowledge file
@st.cache_data
def load_knowledge():
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()
    sentences = text.split("\n")
    return sentences

sentences = load_knowledge()

# Create embeddings
@st.cache_data
def create_embeddings(sentences):
    embeddings = model.encode(sentences)
    return embeddings

embeddings = create_embeddings(sentences)

# User input
query = st.text_input("Ask a question")

if query:
    query_embedding = model.encode([query])

    # Calculate similarity
    similarity = np.dot(embeddings, query_embedding.T)

    best_match = np.argmax(similarity)

    response = sentences[best_match]

    st.write("💬 Answer:")
    st.success(response)
