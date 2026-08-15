import paddlenlp
from paddlenlp.transformers import BertTokenizer

from dataset import PoemData
from model import PoetryBertModel,PoetryBertModelLossCriterion

import paddle
import re

from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity
from paddle.optimizer import AdamW

with open("./data/qilv_all.txt","r") as f:
    out = f.readlines()
out = [i.replace("\n","") for i in out]
out = [i[i.index(":")+1:] for i in out]

lengths = [0.2, 0.2, 0.6]
lengths = [int(0.2*len(out)),int(0.2*len(out)),len(out)-2*int(0.2*len(out))]
test_dataset, dev_dataset, train_dataset = paddle.io.random_split(out,lengths)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


net = PoetryBertModel('bert-base-chinese', 128)#3.3模型准备

token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')
label = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label, input_mask]

model = paddle.Model(net, inputs, labels)
model.prepare(optimizer=AdamW(learning_rate=0.0001, parameters=model.parameters()), loss=PoetryBertModelLossCriterion(), metrics=[Perplexity()])

model.summary(inputs, [input.dtype for input in inputs])


from paddle.io import DataLoader#3.4模型训练

train_loader = DataLoader(PoemData(train_dataset, bert_tokenizer, 128), batch_size=128, shuffle=True)
dev_loader = DataLoader(PoemData(dev_dataset, bert_tokenizer, 128), batch_size=32, shuffle=True)
model.fit(train_data=train_loader, epochs=10, save_dir='./checkpoint7l', save_freq=1, verbose=1, eval_data=dev_loader, eval_freq=1)