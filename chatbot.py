from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge from file
with open("knowledge.txt", "r") as file:
    sentences = file.readlines()

sentences = [s.strip() for s in sentences if s.strip() != ""]

# Convert knowledge to embeddings
embeddings = model.encode(sentences)

print("AI Chatbot Ready! Type 'exit' to stop.")

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        break

    question_embedding = model.encode([user_question])

    similarity = np.dot(embeddings, question_embedding.T)

    index = similarity.argmax()

    answer = sentences[index]

    print("Bot:", answer)