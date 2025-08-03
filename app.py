from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import os
import logging

# Logging
logging.basicConfig(level=logging.INFO)

# API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9mr7-JTjX6ZlHVfGzxRZ1StM2QCBIKCg"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Excel and chunking
df = pd.read_excel("amirfinal.xlsx")
documents = [Document(page_content=" ".join([str(cell) for cell in row if pd.notnull(cell)])) for _, row in df.iterrows()]
chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(documents)

# Embedding and FAISS
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embedding_model)

# LLM + Prompt
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
prompt = PromptTemplate.from_template("""
You are Scene Bot 🎉, a friendly and fun e-commerce assistant for an entertainment platform!
Always sound excited, use emojis, and call the user by their name: {name}.
Use only the dataset and stay relevant to the user's location: {location}.
📚 Context from past chat: {history}
📍 Location-based results: {context}
❓ Question: {input}
💡 Answer:
""")
qa_chain = create_stuff_documents_chain(llm, prompt)

# Sessions
sessions = {}

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/query")
def query(q: Query):
    session_id = q.session_id
    user_input = q.question.lower().strip()

    # New session
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "name": None,
            "location": None,
            "stage": "intro"
        }
        return {"answer": "🎉 Hii! I'm Scene Bot – What's your name? 😊", "reset": True}

    session = sessions[session_id]

    # Intro stage
    if session["stage"] == "intro":
        name = extract_name(user_input)
        session["name"] = name
        session["stage"] = "location"
        return {"answer": f"Awesome, {name}! 😍 Now tell me, which city are you from?"}

    # Location stage
    if session["stage"] == "location":
        location = extract_location(user_input)
        session["location"] = location
        session["stage"] = "chat"
        return {"answer": f"Great! You're from {location}! 🎭 Ask me anything about our entertainment ideas, events, or activities, {session['name']}! 😄"}

    # Chat stage
    location = session["location"].lower()
    filtered_docs = [doc for doc in chunks if location in doc.page_content.lower()]
    top_docs = FAISS.from_documents(filtered_docs, embedding_model).similarity_search(q.question, k=5)

    # Use session memory
    past_history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in session["history"][-3:] if 'user' in h and 'bot' in h])

    reply = qa_chain.invoke({
        "input": q.question,
        "context": top_docs,
        "name": session["name"],
        "location": session["location"],
        "history": past_history
    })

    # Update session
    session["history"].append({"user": q.question, "bot": reply})
    return {"answer": reply}

@app.get("/reset/{session_id}")
def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}

def extract_name(text):
    import re
    match = re.search(r"(?:i am|i'm|im|my name is|this is)?\s*([a-zA-Z]+)", text)
    return match.group(1).capitalize() if match else "Friend"

def extract_location(text):
    words = text.strip().split()
    return words[-1].capitalize() if words else "Somewhere"
