import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertTokenizer

# 加载预训练的BERT模型和tokenizer
model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 自定义一个简单的诗歌生成器模型
class PoetryGenerator(nn.Layer):
    def __init__(self, bert_model):
        super(PoetryGenerator, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(768, tokenizer.vocab_size) # 根据BERT模型输出的隐藏状态大小确定线性层输入和输出大小

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0] # 获取BERT模型输出的隐藏状态
        output = self.linear(output) # 通过线性层映射到词汇表大小的空间
        return output

# 初始化诗歌生成器模型
poetry_generator = PoetryGenerator(model)

# 定义输入文本并编码为input_ids
input_text = "我欲乘风破浪"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids = paddle.to_tensor([input_ids['input_ids']], dtype='int64') # 提取input_ids['input_ids']并转换为张量

# 生成诗歌
output_logits = poetry_generator(input_ids, attention_mask=None) # 不需要添加一维，直接传递input_ids即可
output_ids = paddle.argmax(output_logits, axis=-1).numpy() # 获取概率最高的词的索引
generated_poem = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("生成的诗歌：")
print(generated_poem)
