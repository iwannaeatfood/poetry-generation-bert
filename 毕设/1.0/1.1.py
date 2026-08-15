import paddle
import paddlenlp
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam
from paddlenlp.transformers import BertModel, BertForPretraining
import re

# 定义模型
class PoetryModel(paddle.nn.Layer):
    def __init__(self, bert_model, vocab_size):
        super(PoetryModel, self).__init__()
        self.bert = bert_model
        self.fc = paddle.nn.Linear(self.bert.config["hidden_size"], vocab_size)

    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(input_ids, token_type_ids)
        logits = self.fc(pooled_output)
        return logits

# 数据预处理
def data_preprocess(dataset):
    for i, data in enumerate(dataset):
        dataset.data[i] = ''.join(list(dataset[i].values()))
        dataset.data[i] = re.sub('\\x02', '', dataset[i])
    return dataset

# 加载数据集
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False)
test_dataset = data_preprocess(test_dataset)
dev_dataset = data_preprocess(dev_dataset)
train_dataset = data_preprocess(train_dataset)

#########
# 创建词汇表
vocab = set()
for data in train_dataset:
    vocab.update(data)
vocab = list(vocab)

vocab_size = len(vocab)

bert_model = BertModel.from_pretrained('bert-base-chinese')
model = PoetryModel(bert_model, vocab_size)



# 构建模型
vocab_size = len(vocab)  # 假设你已经定义了词汇表 vocab
bert_model = BertModel.from_pretrained('bert-base-chinese')
model = PoetryModel(bert_model, vocab_size)


# 定义损失函数和优化器
loss_fn = CrossEntropyLoss()
optimizer = Adam(parameters=model.parameters())

# 训练模型
num_epochs = 10  # 你可以设置一个合适的epoch数
for epoch in range(num_epochs):
    for batch in train_dataloader:  # 假设你已经定义了数据加载器 train_dataloader
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))

# 评估模型
model.eval()
correct = 0
total = 0
with paddle.no_grad():
    for batch in test_dataloader:  # 假设你已经定义了数据加载器 test_dataloader
        inputs, targets = batch
        outputs = model(inputs)
        _, predicted = paddle.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# 使用模型进行预测
model.eval()
start_text = '白日依山尽'  # 你的诗的开头
generated_text = start_text
with paddle.no_grad():
    for _ in range(100):  # 生成100个字符
        inputs = paddle.to_tensor([char_to_id[c] for c in start_text])  # 假设你已经定义了字符到ID的映射 char_to_id
        outputs = model(inputs)
        _, predicted = paddle.max(outputs.data, 1)
        generated_text += id_to_char[predicted[-1].item()]  # 假设你已经定义了ID到字符的映射 id_to_char
        start_text = start_text[1:] + id_to_char[predicted[-1].item()]

print('Generated text: ', generated_text)
