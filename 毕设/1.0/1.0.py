import paddle
import paddlenlp 
import re 
from paddle.static import InputSpec
from paddlenlp.transformers import BertModel, BertTokenizer,BertForPretraining


test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False) 
print('test_dataset 的样本数量：%d'%len(test_dataset)) 
print('dev_dataset 的样本数量：%d'%len(dev_dataset)) 
print('train_dataset 的样本数量：%d'%len(train_dataset))

# 定义模型
class PoetryBertModel(BertForPretraining):
    def __init__(self, bert_model):
        super(PoetryBertModel, self).__init__(bert_model)
        self.bert_model = bert_model
        self.linear = paddle.nn.Linear(self.bert_model.config["hidden_size"], self.bert_model.config["vocab_size"])
        self.activation = paddle.nn.functional.softmax
     
    def forward(self, input_ids, token_type_ids=None):
        sequence_output, pooled_output = self.bert_model(input_ids, token_type_ids)         
        prediction_scores = self.linear(sequence_output)         
        prediction_scores = self.activation(prediction_scores)         
        return prediction_scores  
    
# 数据预处理 
def data_preprocess(dataset):     
    for i, data in enumerate(dataset):         
        dataset.data[i] = ''.join(list(dataset[i].values()))         
        dataset.data[i] = re.sub('\\x02', '', dataset[i])     
    return dataset  
# 模型评估 
def evaluate(model, data_loader):     
    model.eval()     
    total_loss = 0     
    total_correct = 0     
    total_num = 0     
    for batch in data_loader:         
        input_ids, labels = tokenizer(batch)         
        logits = model(input_ids)         
        loss = paddle.nn.functional.cross_entropy(logits, labels)         
        total_loss += loss.numpy().item()         
        total_correct += (logits.argmax(-1) == labels).sum().numpy().item()         
        total_num += labels.numel()     
    model.train()     
    return total_loss / total_num, total_correct / total_num  
    
# 生成诗歌 
def generate_poem(model, start_token, length=64):     
    model.eval()     
    input_ids = tokenizer(start_token)['input_ids']
    for _ in range(length):
        logits = model(input_ids)         
        next_token_id = logits[0, -1, :].argmax(-1).unsqueeze(0)
        input_ids = paddle.concat([input_ids, next_token_id], axis=-1)     
    poem = tokenizer.convert_ids_to_tokens(input_ids.numpy().tolist()[0])     
    return ''.join(poem)  
        
# 加载数据集 
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False) 
test_dataset = data_preprocess(test_dataset) 
dev_dataset = data_preprocess(dev_dataset)
train_dataset = data_preprocess(train_dataset)  
    
# 定义模型和tokenizer 
model = PoetryBertModel(BertModel.from_pretrained('bert-base-chinese')) 
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  

# 定义优化器和数据读取器 
optimizer = paddle.optimizer.Adam(parameters=model.parameters())
data_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)  

# 开始训练 
for epoch in range(10):     
    for batch in data_loader:       
        input_ids, labels = tokenizer(batch)         
        logits = model(input_ids)     
        loss = paddle.nn.functional.cross_entropy(logits, labels)    
        loss.backward()         
        optimizer.step()     
        optimizer.clear_grad()    
        
# 在每个epoch结束后在验证集上进行评估
val_loss, val_acc = evaluate(model, dev_data_loader) 
print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')  

# 保存模型参数
paddle.save(model.state_dict(), 'poetry_bert_model.pdparams')  

# 生成一首诗
poem = generate_poem(model, '白日依山尽')
print(poem)