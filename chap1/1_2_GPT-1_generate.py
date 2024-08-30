from transformers import AutoTokenizer, OpenAIGPTLMHeadModel
tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = OpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")
input_ids = tokenizer.encode("A long time ago, in a galaxy far", return_tensors="pt")
outputs = model.generate(input_ids, max_length=42, do_sample=True)
text_generated = tokenizer.decode(outputs[0])
print(text_generated)