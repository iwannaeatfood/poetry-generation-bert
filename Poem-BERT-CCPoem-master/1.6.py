import paddle
from paddlenlp.transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 定义生成古诗的函数
def generate_poetry(start_text, max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(start_text)
    print("Input IDs before conversion to tensor:", input_ids)
    input_ids_tensor = paddle.to_tensor([input_ids])
    print("Input tensor shape:", input_ids_tensor.shape)
    for _ in range(max_length):
        input_ids_tensor = paddle.to_tensor([input_ids])
        # 将输入传入模型得到预测结果
        output = model(input_ids_tensor)
        # 获取预测的下一个字的概率分布
        logits = output['logits']
        next_token_logits = logits[0][-1]
        # 根据temperature调节预测的概率分布
        next_token_logits = next_token_logits / temperature
        # 使用softmax函数计算归一化的概率分布
        next_token_probs = paddle.nn.functional.softmax(next_token_logits, axis=-1)
        # 根据概率分布随机采样得到下一个字的id
        next_token_id = paddle.multinomial(next_token_probs, num_samples=1).numpy()[0][0]
        # 将预测的下一个字加入到输入文本中，准备下一轮预测
        input_ids.append(next_token_id)
        # 如果预测到终止符号，则结束生成
        if next_token_id == tokenizer.sep_token_id:
            break
    # 将生成的文本转换为字符串
    generated_text = tokenizer.decode(input_ids)
    return generated_text

# 生成古诗示例
start_text = "床前明月光，"
generated_poetry = generate_poetry(start_text, max_length=50, temperature=0.7)
print("生成的古诗：", generated_poetry)
