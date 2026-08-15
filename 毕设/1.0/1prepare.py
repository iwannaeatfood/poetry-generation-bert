import paddle 
import paddlenlp
import re  
# 安装飞桨2.0和PaddleNLP !pip install paddlepaddle paddlenlp  

#加载数据集 
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False) 
# 数据预处理 
def data_preprocess(dataset):    
    for i, data in enumerate(dataset):       
        dataset.data[i] = ''.join(list(dataset[i].values()))   
        dataset.data[i] = re.sub('\\x02', '', dataset[i])   
    return dataset 
test_dataset = data_preprocess(test_dataset) 
dev_dataset = data_preprocess(dev_dataset)
train_dataset = data_preprocess(train_dataset) 