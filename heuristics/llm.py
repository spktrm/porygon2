from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("crumb/nano-mistral")
tokenizer = AutoTokenizer.from_pretrained("crumb/nano-mistral")

inputs = tokenizer(["Once upon a time,"], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in dict(inputs).items()}
outputs = model.generate(
    inputs, max_new_tokens=128, temperature=0.7, top_k=20, do_sample=True
)
outputs = tokenizer.batch_decode(outputs)
for i in outputs:
    print(i)
