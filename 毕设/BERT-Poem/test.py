from model import PoetryBertModel
from gen import PoetryGen
from paddlenlp.transformers import BertTokenizer

import re
import paddle
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 载入已经训练好的模型
net = PoetryBertModel('bert-base-chinese', 128)
model = paddle.Model(net)
model.load('./checkpoint5j/checkpoint5j/final')
poetry_gen = PoetryGen(model, bert_tokenizer,poetry_len=5,poetry_type="j")
def poetry_show(poetry):
    pattern = r"([，。；？])"
    text = re.sub(pattern, r'\1 ', poetry)
    for p in text.split():
        if p:
            print(p)
# 随机生成一首诗
poetry = poetry_gen.generate()
poetry_show(poetry)
