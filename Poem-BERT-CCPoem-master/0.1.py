import paddlenlp as ppnlp
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300") # 一键加载中文词向量
print(wordemb.cosine_sim("国王", "王后"))
print(wordemb.cosine_sim("艺术", "火车"))

test_dataset, dev_dataset, train_dataset = ppnlp.datasets.load_dataset('poetry', splits=('test', 'dev', 'train'), lazy=False)
print('test_dataset 的样本数量：%d' % len(test_dataset))
print('dev_dataset 的样本数量：%d' % len(dev_dataset))
print('train_dataset 的样本数量：%d' % len(train_dataset))
