from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载GPT-2预训练模型和tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 输入一句话，例如：我欲乘风破浪
input_text = "我欲乘风破浪"
input_ids = tokenizer.encode(input_text, return_tensors='pt')  # 返回PyTorch张量形式的编码结果

# 生成古诗
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_poem = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的古诗：")
print(generated_poem)
