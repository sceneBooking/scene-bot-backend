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
import re

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
df = pd.read_excel("DATASET.xlsx")
documents = [Document(page_content=" ".join([str(cell) for cell in row if pd.notnull(cell)])) for _, row in df.iterrows()]
chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(documents)

# Embedding and FAISS
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_documents(chunks, embedding_model)

# LLM + Prompt
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)
prompt = PromptTemplate.from_template("""
You are Scene Bot 🎉, a friendly booking assistant for Scene Booking platform!
Scene Booking is a platform for activities, workshops, and events across Chennai, Hyderabad, Coimbatore, and Pondicherry.
Always sound joyful 😍, use emojis 🎭, and call the user by their name: {name}.

Strict rules:
- ANSWER VERY EXCITINGLY WITH 2 to 4 RELEVENT EMOJIES per answer(use emojies LIKE 🎉 🤩 😍 🥳 😄 😊 😎 🤗 🙌 ✨ 🌟 💫 🎭 🎪 🎶 🎵 🎤 🎧 🎷 🎸 🎹 🥁 🎻 🎯 🎨 🖌️ 🖼️ ✏️ 📝 🖊️ 🖋️ ✒️ 🧵 🪡 🧶 🪢 🧩 🪆 ✂️ 🪞 🪵 🧱 🍕 🍔 🍩 🍰 🧁 🎂 🍪 🍫 🍬 🍭 🍦 🍨 🥂 🍷 ☕ 🍵 🍹 🍸 🍺 ❤️ 💕 💞 💖 💘 💝 💓 🌹 🌺 🌸 🌻 🌼 🕯️ 🌙 🌌 💍 🎲 🎮 🎳 ⚽ 🏀 🏏 🥊 🏋️ 🤹 🎢 🎡 🎠 🚴 🛶 🧗 🏕️ 🏖️ 🪂 🎬 📽️ 🎞️ 🍿 🎟️ 🎫 📍 🗺️ 🏙️ 🌆 🏖️ 🏞️ 🌄 🏛️ 🌉 🌌 🏝️ 🌇 🌃 💰 💵 💳 🪙 🛒 🛍️ 📅 📌 📝 ✍️ 📲).
- Only use dataset context.
- When listing multiple activities, ALWAYS format like this:

There are X activities in {location}:

1. Activity Name (₹Price)
2. Activity Name (₹Price)
...

- Do not use *, **, or markdown. Always return clean numbered lists or friendly sentences.
- If you include URLs, always make them clickable links in HTML: <a href="URL" target="_blank">Link Text</a>

📚 Past Chat Context:
{history}

📍 Location-based Results:
{context}

❓ User Question:
{input}

💡 Answer:
""")
qa_chain = create_stuff_documents_chain(llm, prompt)

# Sessions
sessions = {}

class Query(BaseModel):
    question: str
    session_id: str

def make_links_clickable(text):
    """Convert URLs in text to clickable HTML links, avoiding double-processing"""
    if '<a href=' in text:
        return text
    url_pattern = r'(?<!href=")(?<!href=\')(?<!>)(https?://[^\s<>"]+|www\.[^\s<>"]+)(?!</a>)'
    def replace_url(match):
        url = match.group(1)
        full_url = 'https://' + url if url.startswith('www.') else url
        return f'<a href="{full_url}" target="_blank">{url}</a>'
    return re.sub(url_pattern, replace_url, text)


def format_structured_answer(text):
    """
    Ensure activities are displayed cleanly one by one like ChatGPT.
    Example:
    There are 23 activities in Anna Nagar:

    1. Resin Art on Mirror (₹1499)
    2. Archery (₹400)
    ...
    """
    # Extract activity-like patterns: anything with ₹price or ending with "Workshop/Party/Class"
    activity_pattern = r'([A-Za-z0-9 ,\-–&]+?\(₹[0-9,]+(?:\s*₹[0-9,]+)?\))'
    activities = re.findall(activity_pattern, text)

    # If multiple activities found, reformat them into a clean numbered list
    if len(activities) > 1:
        formatted = f"There are {len(activities)} activities in this area:<br><br>"
        for i, act in enumerate(activities, start=1):
            formatted += f"{i}. {act.strip()}<br>"
        return formatted.strip()

    # If no multiple matches, return original text
    return text


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
    if not filtered_docs:
        filtered_docs = chunks

    top_docs = FAISS.from_documents(filtered_docs, embedding_model).similarity_search(q.question, k=5)

    # Full past history
    past_history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in session["history"]])

    reply = qa_chain.invoke({
        "input": q.question,
        "context": top_docs,
        "name": session["name"],
        "location": session["location"],
        "history": past_history
    })

    # Format reply
    reply = format_structured_answer(str(reply))
    reply = make_links_clickable(reply)

    # Save history
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
