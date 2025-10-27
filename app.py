from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model

app = FastAPI(title="GPT-20B API")

# Request modeli
class GenerateRequest(BaseModel):
    prompt: str

# Modelni yuklaymiz
model = load_model()

@app.get("/")
def home():
    return {"message": "GPT-20B API is running!"}

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        output = model(
            request.prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return {"response": output[0]["generated_text"]}
    except Exception as e:
        return {"error": str(e)}
