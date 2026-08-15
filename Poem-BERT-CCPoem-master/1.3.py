import os
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

gpu_list = None

# 定义一个名为 Bert 的类，继承自 nn.Module
class Bert(nn.Module):
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        super(Bert, self).__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(BERT_PATH)

    # 初始化多 GPU 训练
    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    # 前向传播函数，接受输入数据 data 和是否只输出 CLS token 的标志 cls
    def forward(self, data, cls=False):
        result = []
        # 获取输入数据的 input_ids
        x = data['input_ids']
        # 通过 BERT 模型获取输出
        y = self.bert(x, attention_mask=data['attention_mask'],
                      token_type_ids=data['token_type_ids'])[0]

        # 如果只输出 CLS token
        if (cls):
            result = y[:, 0, :].view(y.size(0), -1)
            # 将结果转换为 CPU 上的张量并转换为列表
            result = result.cpu().tolist()
        else:
            result = []
            y = y.cpu()
            # 遍历每个样本的输出
            for i in range(y.shape[0]):
                # 获取有效 token 的输出并取平均值
                tmp = y[i][1:torch.sum(data['attention_mask'][i]) - 1, :]
                result.append(tmp.mean(0).tolist())

        return result


# 定义一个名为 BertFormatter 的类
class BertFormatter():
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        # 加载 BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 数据预处理函数，将输入数据转换为模型可以接受的形式
    def process(self, data):
        # 使用 tokenizer 进行批量编码
        res_dict = self.tokenizer.batch_encode_plus(
            data, pad_to_max_length=True)

        # 将编码结果转换为 PyTorch 张量格式
        input_list = {'input_ids': torch.LongTensor(res_dict['input_ids']),
                      'attention_mask': torch.LongTensor(res_dict['attention_mask']),
                      "token_type_ids": torch.LongTensor(res_dict['token_type_ids'])}
        return input_list


# 初始化模型和数据处理器
def init(BERT_PATH="./BERT_CCPoem_v1"):
    global gpu_list
    gpu_list = []

    # 获取可用的 GPU 列表
    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (device_list[0] == ""):
        device_list = []
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logging.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logging.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    # 创建 Bert 模型和数据处理器实例
    model = Bert(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    if len(gpu_list) > 0:
        model = model.cuda()
    if (len(gpu_list) > 1):
        try:
            # 初始化多 GPU 训练
            model.init_multi_gpu(gpu_list)
        except Exception as e:
            logging.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead. {}".format(str(e)))
    return model, formatter


# 预测向量表示
def predict_vec_rep(data, model, formatter):
    data = formatter.process(data)
    # 将模型设置为评估模式
    model.eval()

    for i in data:
        if (isinstance(data[i], torch.Tensor)):
            if len(gpu_list) > 0:
                # 如果 GPU 可用，将数据转移到 GPU 上
                data[i] = data[i].cuda()

    # 获取模型的输出
    result = model(data)

    return result


if __name__ == '__main__':
    # 初始化模型和数据处理器
    model, formatter = init()

    # 生成诗歌示例
    result = predict_vec_rep(["一行白鹭上青天"], model, formatter)[0]
    print(result)



def init(BERT_PATH="./BERT_CCPoem_v1"):
    model = Bert(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    return model, formatter
