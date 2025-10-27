import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model():
    torch.cuda.empty_cache()
    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return pipe

