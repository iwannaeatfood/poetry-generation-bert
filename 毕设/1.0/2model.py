import paddle 
from paddlenlp.transformers import BertModel, BertForPretraining 
# 定义模型
class PoetryModel(paddle.nn.Layer):     
    def __init__(self, vocab_size):       
        super(PoetryModel, self).__init__()      
        self.bert = BertModel.from_pretrained('bert-base-chinese')         
        self.fc = paddle.nn.Linear(self.bert.config["hidden_size"], vocab_size)  
    def forward(self, input_ids, token_type_ids):         
        _, pooled_output = self.bert(input_ids, token_type_ids)     
        logits = self.fc(pooled_output)        
        return logits 
vocab_size = len(vocab)  
# 假设你已经定义了词汇表vocab 
model = PoetryModel(vocab_size) 