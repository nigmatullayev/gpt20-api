from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_model():
    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Majburan GPU0ga chiqaramiz
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
    )
    return pipe
