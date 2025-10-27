from fastapi import FastAPI
from model import load_model

app = FastAPI()
model = load_model()

@app.get("/")
def home():
    return {"message": "GPT-20B API is running!"}

@app.post("/generate")
def generate(prompt: str):
    output = model(prompt, max_new_tokens=200)
    return {"response": output[0]["generated_text"]}
