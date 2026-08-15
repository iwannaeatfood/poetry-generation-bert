import warnings

from transformers import BertModel, BertTokenizer
import torch

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('./BERT_CCPoem_v1')
model = BertModel.from_pretrained('./BERT_CCPoem_v1')
model.eval()
input_ids = torch.tensor(tokenizer.encode("一行白鹭上青天")).unsqueeze(0)
result = model(input_ids)
sen_emb = torch.mean(result['last_hidden_state'], 1)[0]

print(sen_emb)



