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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        "Halo! Ada yang bisa saya bantu soal keuangan? 😊",
        "Hai! Saya siap bantu soal finansial kamu. 👋",
        "Halo! Tanya aja soal keuangan, saya siap bantu. 💬",
    ],

    "expense": [
        "💸 Pengeluaran kecil sekalipun perlu dicatat — lama-lama bisa jadi besar.",
        "📉 Coba cek pengeluaran terbesar bulan ini, biasanya di situ potensi hemat paling banyak.",
        "💡 Pisahkan pengeluaran wajib dan opsional supaya lebih mudah dikontrol.",
    ],

    "budgeting": [
        "📊 Coba metode 50/30/20 — 50% kebutuhan, 30% keinginan, 20% tabungan.",
        "🎯 Budget yang baik bukan soal pelit, tapi soal tahu uang kamu pergi ke mana.",
        "💡 Catat dulu semua pengeluaran selama seminggu — biasanya hasilnya mengejutkan.",
    ],

    "investment": [
        "📈 Investasi paling penting: sesuaikan dengan profil risiko dan tujuan kamu.",
        "💰 Mulai dari yang kamu pahami dulu — reksa dana pasar uang bagus untuk pemula.",
        "🎯 Diversifikasi itu kunci — jangan taruh semua di satu instrumen.",
    ],

    "fraud": [
        "⚠️ Jangan pernah bagikan OTP, PIN, atau password ke siapapun — termasuk yang mengaku pihak bank.",
        "🔒 Kalau ada transaksi mencurigakan, segera blokir kartu dan hubungi bank kamu.",
        "🚨 Waspadai tawaran investasi dengan imbal hasil tidak masuk akal — itu ciri-ciri scam.",
    ],

    "saving": [
        "🏦 Simpan dulu di awal bulan, bukan sisanya — bayar diri sendiri terlebih dahulu.",
        "💡 Target tabungan yang realistis lebih baik daripada target besar tapi tidak konsisten.",
        "📅 Otomatisasi tabungan lewat auto-debit supaya tidak tergoda pakai duluan.",
    ],

    "income": [
        "📈 Tingkatkan income dengan skill atau side hustle — pengeluaran ada batasnya, income tidak.",
        "💼 Diversifikasi sumber penghasilan bisa jadi jaring pengaman finansial yang kuat.",
    ],
}

# =========================
# GROQ FUNCTION
# =========================

def ask_groq(query: str) -> str:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten keuangan pribadi yang cerdas dan ringkas. "
                    "Jawab dalam Bahasa Indonesia, maksimal 2-3 kalimat, langsung ke inti. "
                    "Gunakan emoji yang relevan (💸📈📉💡🎯🏦⚠️) untuk membuat jawaban lebih hidup. "
                    "Jangan buat daftar panjang. Jika pertanyaan tidak jelas, minta klarifikasi singkat."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return completion.choices[0].message.content.strip()

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

    pred  = intent_pipeline.predict([query])[0]
    proba = max(intent_pipeline.predict_proba([query])[0])

    # =====================
    # HYBRID ROUTING
    # =====================

    # Intent yang bisa dijawab lokal (saran murni, tidak butuh data user)
    LOCAL_INTENTS = {"budgeting", "investment", "fraud"}

    if proba >= 0.55 and pred in LOCAL_INTENTS:
        answer = random.choice(responses[pred])
        source = "LOCAL"

    else:
        # Semua selain itu → Groq (termasuk expense, income, saving, general, dll)
        answer = ask_groq(query)
        source = "API_GROQ"

    # =====================
    # RESPONSE
    # =====================

    return {
        "query":      query,
        "intent":     pred,
        "confidence": round(float(proba), 2),
        "source":     source,
        "answer":     answer,
    }
