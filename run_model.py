# run_model_gpu.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_model():
    model_id = "openai/gpt-oss-20b"

    if not torch.cuda.is_available():
        raise RuntimeError("GPU topilmadi! Ushbu skript faqat GPUda ishlaydi.")

    print("Model va tokenizer yuklanmoqda... Bu GPUda biroz vaqt oladi.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # GPUda quantized modelni yuklash
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",       # avtomatik GPU device map
        torch_dtype=torch.bfloat16,  # bf16 GPUda tez ishlaydi
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        device=0  # GPU0
    )
    return pipe

if __name__ == "__main__":
    pipe = load_model()

    print("\nModel GPUda tayyor! Terminalda prompt yozing (CTRL+C bilan chiqish):\n")
    while True:
        try:
            prompt = input(">>> ")
            output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
            print("\n" + output[0]["generated_text"] + "\n")
        except KeyboardInterrupt:
            print("\nChiqish...")
            break
        except Exception as e:
            print("Xatolik:", e)
