import paddle
import paddlenlp
from paddle.io import DataLoader, Dataset
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import Adam
from paddlenlp.transformers import BertModel, BertForPretraining
import re

# 定义模型
class PoetryModel(paddle.nn.Layer):
    def __init__(self, bert_model, vocab_size):
        super(PoetryModel, self).__init__()
        self.bert = bert_model  # 使用预训练的BERT模型
        self.fc = paddle.nn.Linear(self.bert.config["hidden_size"], vocab_size)  # 添加一个全连接层，用于生成诗句

    def forward(self, input_ids):
        _, pooled_output = self.bert(input_ids)  # 通过BERT模型得到每个输入的表示
        logits = self.fc(pooled_output)  # 通过全连接层得到最终的输出
        return logits

# 数据预处理
def data_preprocess(dataset):
    for i, data in enumerate(dataset):
        dataset.data[i] = ''.join(list(dataset[i].values()))  # 将数据集中的每个样本转换为字符串
        dataset.data[i] = re.sub('\\x02', '', dataset[i])  # 去除字符串中的特殊字符
    return dataset

# 加载数据集
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False)
test_dataset = data_preprocess(test_dataset)  # 对测试集进行预处理
dev_dataset = data_preprocess(dev_dataset)  # 对验证集进行预处理
train_dataset = data_preprocess(train_dataset)  # 对训练集进行预处理

# 创建词汇表
vocab = set()
for data in train_dataset:
    vocab.update(data)  # 将训练集中的每个字符添加到词汇表中
vocab = list(vocab)

# 创建字符到ID的映射和一个ID到字符的映射
char_to_id = {char: i for i, char in enumerate(vocab)}  # 创建字符到ID的映射
id_to_char = {i: char for i, char in enumerate(vocab)}  # 创建ID到字符的映射

# 数据加载器
class PoetryDataset(Dataset):
    def __init__(self, data, char_to_id):
        self.data = data  # 数据集
        self.char_to_id = char_to_id  # 字符到ID的映射

    def __getitem__(self, idx):
        return self.data[idx]  # 返回指定索引的样本

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

train_dataset = PoetryDataset(train_dataset, char_to_id)  # 创建训练数据集
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建数据加载器

# 初始化模型和优化器
bert_model = BertModel.from_pretrained('bert-base-chinese')  # 加载预训练的BERT模型
model = PoetryModel(bert_model, len(vocab))  # 创建模型
loss_fn = CrossEntropyLoss()  # 定义损失函数
optimizer = Adam(parameters=model.parameters())  # 定义优化器

# 训练模型
num_epochs = 10  # 设置训练的轮数
for epoch in range(num_epochs):
    for batch in train_dataloader:  # 对于训练数据集中的每个批次
        inputs = paddle.to_tensor([char_to_id[c] for c in batch], dtype='int64').unsqueeze(0)
                # 将输入转换为模型可以接受的形式
        outputs = model(inputs)  # 将输入传入模型，得到输出
        loss = loss_fn(outputs, inputs)  # 计算损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型的参数
        optimizer.clear_grad()  # 清除梯度

        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))  # 打印每轮训练的损失

# # 使用模型进行预测
model.eval() #将模型设置为评估模式
start_text = '白日依山尽'
#设置预测的开始文本
generated_text = start_text #初始化生成的文本
with paddle.no_grad(): #不计算梯度
    for _ in range(100):#生成100个字符
        inputs = paddle.to_tensor([char_to_id[c] for c in start_text],dtype='int64').unsqueeze(8)# 将输入转换为模型可以接受的形式
        outputs = model(inputs)# 将输入传入模型，得到输出
        predicted = paddle.argmax(outputs,axis=-1).numpy()[0] #得到预测的字符的
        generated_text += id_to_char[predicted[-1]] #将预测的字符添加到生成的文本中
        start_text = start_text[1:]+ id_to_char[predicted[-1]] #更新输入
print('Generated text:', generated_text)#打印生成的文本
