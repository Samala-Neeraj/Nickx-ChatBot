import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder containing documents
folder_path = "documents"

chunks = []

# Read all text files
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            parts = text.split("\n")
            chunks.extend(parts)

# Create embeddings
chunk_embeddings = model.encode(chunks)

st.title("Nick'x 🤖")

user_input = st.text_input("Ask a question")

if user_input:

    # -------- Name Detection --------
    match = re.search(r"(i am|i'm|my name is)\s+(\w+)", user_input.lower())

    if match:
        name = match.group(2).capitalize()
        st.write(f"Nick'x: Hi {name}, I'm Nick'x. How may I help you?")
    
    else:

        # -------- Normal Chatbot Logic --------
        question_embedding = model.encode([user_input])

        similarities = np.dot(chunk_embeddings, question_embedding.T)

        best_match = np.argmax(similarities)

        score = similarities[best_match][0]

        if score > 0.3:
            st.write("Nick'x:", chunks[best_match])
        else:
            st.write("Nick'x: Sorry, I couldn't find an answer.")