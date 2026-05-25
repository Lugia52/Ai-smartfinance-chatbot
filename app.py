from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from groq import Groq

import pickle
import random
import os

# =========================
# LOAD MODEL
# =========================

with open("intent_pipeline.pkl", "rb") as f:
    intent_pipeline = pickle.load(f)

# =========================
# GROQ CLIENT
# =========================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    message: str

# =========================
# LOCAL RESPONSES
# =========================

responses = {

    "general": [
        "Halo! Saya Smart Finance Chatbot.",
        "Hai! Ada yang bisa saya bantu soal finansial?"
    ],

    "expense": [
        "Pengeluaran tetap perlu dicatat agar budget terkontrol.",
        "Pengeluaran kecil tetap perlu dimonitor."
    ],

    "budgeting": [
        "Pisahkan kebutuhan dan keinginan saat budgeting."
    ],

    "investment": [
        "Investasi sebaiknya sesuai profil risiko."
    ],

    "fraud": [
        "Waspadai transaksi mencurigakan dan jangan bagikan OTP."
    ]
}

# =========================
# GROQ FUNCTION
# =========================

def ask_groq(query):

    completion = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content":
                "Kamu adalah Smart Finance Chatbot yang membantu soal keuangan pribadi."
            },

            {
                "role": "user",
                "content": query
            }
        ]
    )

    return completion.choices[0].message.content

# =========================
# ROOT
# =========================

@app.get("/")
def home():
    return FileResponse("index.html")

# =========================
# CHAT
# =========================

@app.post("/chat")
def chat(req: ChatRequest):

    query = req.message

    # prediksi intent
    pred = intent_pipeline.predict([query])[0]

    # confidence
    proba = max(
        intent_pipeline.predict_proba([query])[0]
    )

    # =====================
    # HYBRID ROUTING
    # =====================

    if proba >= 0.55:

        answer = random.choice(
            responses.get(
                pred,
                ["Maaf saya belum memahami."]
            )
        )

        source = "LOCAL"

    else:

        answer = ask_groq(query)

        source = "API_GROQ"

    # =====================
    # RESPONSE
    # =====================

    return {

        "query": query,

        "intent": pred,

        "confidence": float(proba),

        "source": source,

        "answer": answer
    }
