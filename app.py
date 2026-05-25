from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# load model
with open("trained_model.pkl", "rb") as f:
    intent_pipeline = pickle.load(f)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Smart Finance API Running"}

@app.post("/chat")
def chat(req: ChatRequest):

    query = req.message

    # contoh intent
    intent = intent_pipeline.predict([query])[0]

    response = {
        "query": query,
        "intent": intent,
        "answer": f"Detected intent: {intent}"
    }

    return response