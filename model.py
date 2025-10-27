from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model():
    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",           # BF16 / FP16 ishlatadi
        device_map="auto",            # GPUga joylaydi
        trust_remote_code=True        # GPT-OSS uchun kerak
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
    )
    return pipe
