import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Nick'x AI ChatBot", page_icon="🤖")

st.title("🤖 Nick'x AI ChatBot")

# Load model once
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# Load knowledge base
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

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me something...")

if prompt:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert query to embedding
    query_embedding = model.encode([prompt])

    # Calculate similarity
    similarity = np.dot(embeddings, query_embedding.T)

    best_match = np.argmax(similarity)

    response = sentences[best_match]

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
