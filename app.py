from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
import os
import pandas as pd
from groq import Groq
from functools import lru_cache

app = FastAPI()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")
GROQ_MODEL   = "llama3-8b-8192"

# Path CSV data (sesuaikan jika beda lokasi)
MONTHLY_FEATURES_PATH = os.getenv("MONTHLY_FEATURES_PATH", "user_monthly_features.csv")
TRANSACTIONS_PATH     = os.getenv("TRANSACTIONS_PATH",     "transactions_clean.csv")

# Threshold confidence intent — di bawah ini langsung fallback ke Groq
INTENT_CONFIDENCE_THRESHOLD = 0.55

groq_client = Groq(api_key=GROQ_API_KEY)

# ─── LOAD DATA CSV ────────────────────────────────────────────────────────────
try:
    df_monthly = pd.read_csv(MONTHLY_FEATURES_PATH, parse_dates=["period"])
    df_tx      = pd.read_csv(TRANSACTIONS_PATH)
    print(f"✅ Data loaded: {len(df_monthly)} monthly records, {len(df_tx)} transactions.")
except FileNotFoundError as e:
    print(f"⚠️  CSV tidak ditemukan: {e}")
    df_monthly = pd.DataFrame()
    df_tx      = pd.DataFrame()

# ─── LOAD INTENT MODEL ────────────────────────────────────────────────────────
intent_pipeline = None
try:
    with open("trained_model.pkl", "rb") as f:
        intent_pipeline = pickle.load(f)
    print("✅ Intent model loaded.")
except FileNotFoundError:
    print("⚠️  trained_model.pkl tidak ditemukan — semua request akan pakai Groq.")


# ─── MODELS ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:  str
    user_id:  Optional[str] = "guest"
    period:   Optional[str] = None   # format: "2026-05" — default bulan terakhir


# ─── DATA HELPERS ─────────────────────────────────────────────────────────────
def get_user_monthly(user_id: str, period: Optional[str] = None) -> Optional[dict]:
    """Ambil data fitur bulanan user. Kalau period None → pakai bulan terbaru."""
    if df_monthly.empty:
        return None
    rows = df_monthly[df_monthly["user_id"] == user_id]
    if rows.empty:
        return None
    if period:
        rows = rows[rows["period"].dt.strftime("%Y-%m") == period]
    else:
        rows = rows.sort_values("period", ascending=False)
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def get_recent_transactions(user_id: str, n: int = 5, period: Optional[str] = None) -> list:
    """Ambil transaksi terakhir user, opsional filter per period."""
    if df_tx.empty:
        return []
    rows = df_tx[df_tx["user_id"] == user_id].copy()
    if period:
        rows = rows[rows["period"] == period]
    rows = rows.sort_values("transaction_date", ascending=False)
    return rows.head(n)[["description", "amount", "type", "category_name", "transaction_date"]].to_dict("records")


def get_top_categories(user_id: str, period: Optional[str] = None) -> list:
    """Hitung top 3 kategori pengeluaran user berdasarkan total amount."""
    if df_tx.empty:
        return []
    rows = df_tx[(df_tx["user_id"] == user_id) & (df_tx["type"] == "expense")].copy()
    if period:
        rows = rows[rows["period"] == period]
    if rows.empty:
        return []
    top = (
        rows.groupby("category_name")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    return [{"category": k, "total": v} for k, v in top.items()]


def fetch_user_data(user_id: str, period: Optional[str] = None) -> dict:
    """Gabungkan semua data user dari CSV menjadi satu dict konteks."""
    if user_id == "guest":
        return {}

    monthly = get_user_monthly(user_id, period)
    if monthly is None:
        return {}

    recent_tx  = get_recent_transactions(user_id, n=5, period=period)
    top_cats   = get_top_categories(user_id, period=period)

    # Tentukan periode aktual yang dipakai
    actual_period = pd.Timestamp(monthly["period"]).strftime("%B %Y") if pd.notna(monthly.get("period")) else "-"

    return {
        "period":              actual_period,
        "total_income":        monthly.get("total_income"),
        "total_expense":       monthly.get("total_expense"),
        "net_cashflow":        monthly.get("net_cashflow"),
        "saving_rate":         monthly.get("saving_rate"),
        "tx_count":            monthly.get("tx_count"),
        "avg_expense":         monthly.get("avg_expense"),
        "expense_trend":       monthly.get("expense_trend"),   # positif = naik vs bulan lalu
        "rolling_3m_avg":      monthly.get("rolling_3m_avg"),
        # Rasio kategori (0–1)
        "food_ratio":          monthly.get("food_ratio"),
        "transport_ratio":     monthly.get("transport_ratio"),
        "entertainment_ratio": monthly.get("entertainment_ratio"),
        "shopping_ratio":      monthly.get("shopping_ratio"),
        "health_ratio":        monthly.get("health_ratio"),
        "other_ratio":         monthly.get("other_ratio"),
        # Transaksi & kategori
        "top_categories":      top_cats,
        "recent_transactions": recent_tx,
    }


def build_user_context(d: dict) -> str:
    """Ubah dict data user jadi teks ringkas untuk system prompt."""
    if not d:
        return "Data keuangan user tidak tersedia."

    def rp(v):
        return f"Rp {v:,.0f}" if v is not None else "-"

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "-"

    # Trend pengeluaran
    trend = d.get("expense_trend", 0) or 0
    trend_str = f"naik Rp {trend:,.0f}" if trend > 0 else (f"turun Rp {abs(trend):,.0f}" if trend < 0 else "stabil")

    # Top kategori
    top_cats = d.get("top_categories", [])
    cats_str  = ", ".join([f"{c['category']} ({rp(c['total'])})" for c in top_cats]) or "-"

    # Transaksi terakhir
    txs = d.get("recent_transactions", [])
    tx_lines = "\n".join([
        f"  • {t['description']} | {rp(t['amount'])} | {t['category_name']} | {t['transaction_date']}"
        for t in txs
    ]) or "  Tidak ada data"

    return f"""Periode: {d.get('period', '-')}
Pemasukan  : {rp(d.get('total_income'))}
Pengeluaran: {rp(d.get('total_expense'))}
Net cashflow: {rp(d.get('net_cashflow'))}
Saving rate : {pct(d.get('saving_rate'))} | Avg transaksi: {rp(d.get('avg_expense'))} | Jumlah transaksi: {int(d.get('tx_count') or 0)}
Tren pengeluaran vs bulan lalu: {trend_str}
Rolling 3 bulan avg cashflow: {rp(d.get('rolling_3m_avg'))}

Rasio kategori pengeluaran:
  Makanan: {pct(d.get('food_ratio'))} | Transport: {pct(d.get('transport_ratio'))} | Hiburan: {pct(d.get('entertainment_ratio'))}
  Belanja: {pct(d.get('shopping_ratio'))} | Kesehatan: {pct(d.get('health_ratio'))} | Lainnya: {pct(d.get('other_ratio'))}

Top kategori terbesar: {cats_str}

5 Transaksi terakhir:
{tx_lines}"""


# ─── INTENT ───────────────────────────────────────────────────────────────────
def predict_intent_with_confidence(query: str):
    if intent_pipeline is None:
        return None, 0.0
    try:
        proba      = intent_pipeline.predict_proba([query])[0]
        confidence = float(max(proba))
        intent     = intent_pipeline.classes_[proba.argmax()]
        return intent, confidence
    except AttributeError:
        return intent_pipeline.predict([query])[0], 1.0


def handle_intent(intent: str, d: dict) -> Optional[str]:
    """Jawab intent deterministik dari data CSV. Return None → fallback Groq."""
    income  = d.get("total_income")
    expense = d.get("total_expense")
    cashflow = d.get("net_cashflow")
    saving  = d.get("saving_rate")
    period  = d.get("period", "bulan ini")

    if intent == "cek_pemasukan":
        if income:
            return f"📈 Pemasukan kamu {period}: **Rp {income:,.0f}**."

    if intent == "cek_pengeluaran":
        if expense:
            return f"📉 Pengeluaran kamu {period}: **Rp {expense:,.0f}**."

    if intent == "cek_tabungan" or intent == "cek_saving_rate":
        if saving is not None:
            return f"🏦 Saving rate kamu {period}: **{saving*100:.1f}%** (net cashflow: Rp {cashflow:,.0f})."

    if intent == "ringkasan_keuangan":
        if income and expense:
            status = "surplus 🟢" if cashflow >= 0 else "defisit 🔴"
            return (
                f"📊 {period} kamu {status} **Rp {abs(cashflow):,.0f}**. "
                f"Pemasukan Rp {income:,.0f}, pengeluaran Rp {expense:,.0f}, "
                f"saving rate {saving*100:.1f}%."
            )

    if intent == "cek_kategori":
        top = d.get("top_categories", [])
        if top:
            cats = ", ".join([f"{c['category']} (Rp {c['total']:,.0f})" for c in top])
            return f"🗂️ Top pengeluaran {period}: {cats}."

    if intent == "cek_transaksi":
        txs = d.get("recent_transactions", [])
        if txs:
            lines = "\n".join([f"• {t['description']} — Rp {t['amount']:,.0f} ({t['category_name']})" for t in txs[:3]])
            return f"🧾 Transaksi terakhir kamu:\n{lines}"

    return None


# ─── GROQ ─────────────────────────────────────────────────────────────────────
def groq_chat(message: str, user_context: str) -> str:
    system_prompt = f"""Kamu adalah asisten keuangan pribadi yang cerdas dan ringkas.

DATA KEUANGAN USER:
{user_context}

ATURAN MENJAWAB:
- Jawab dalam Bahasa Indonesia
- Maksimal 3-4 kalimat, langsung ke inti
- Gunakan data user di atas untuk memberi insight personal dan spesifik
- Sebutkan angka secara konkret jika relevan
- Jangan buat daftar panjang atau poin-poin bertele-tele
- Jika pertanyaan tidak jelas, minta klarifikasi singkat
- Nada: ramah tapi profesional
- Gunakan emoji yang relevan (💸 pengeluaran, 📈 pemasukan, ⚠️ peringatan, 💡 saran, 🎯 tujuan, 🏦 tabungan, dll)"""

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": message},
        ],
        max_tokens=300,
        temperature=0.5,
    )
    return completion.choices[0].message.content.strip()


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Smart Finance API Running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    query   = req.message.strip()
    user_id = req.user_id
    period  = req.period  # e.g. "2026-05", boleh None

    # 1. Fetch & build context dari CSV
    user_data    = fetch_user_data(user_id, period)
    user_context = build_user_context(user_data)

    # 2. Predict intent
    intent, confidence = predict_intent_with_confidence(query)

    answer = None
    source = "groq"

    # 3. Intent handler jika confidence cukup
    if intent and confidence >= INTENT_CONFIDENCE_THRESHOLD:
        answer = handle_intent(intent, user_data)
        if answer:
            source = "intent"

    # 4. Fallback Groq
    if answer is None:
        try:
            answer = groq_chat(query, user_context)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

    return {
        "query":      query,
        "intent":     intent if source == "intent" else None,
        "confidence": round(confidence, 2) if source == "intent" else None,
        "source":     source,
        "answer":     answer,
    }
