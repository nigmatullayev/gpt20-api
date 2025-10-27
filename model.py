from transformers import pipeline

def load_model():
    model_id = "openai/gpt-oss-20b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return pipe
