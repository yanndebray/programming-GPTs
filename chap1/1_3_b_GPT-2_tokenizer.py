from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
print(tokenizer("Hello world")["input_ids"]) # [15496, 995]
print(tokenizer("hello world")["input_ids"]) # [31373, 995]
print(tokenizer(" Hello world")["input_ids"]) # [18435, 995]