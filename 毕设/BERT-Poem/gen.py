import numpy as np  # 导入NumPy库，用于数值计算
import paddle  # 导入PaddlePaddle深度学习框架


# 定义一个诗歌生成类#4.古诗生成
class PoetryGen(object):
    """
    定义一个自动生成诗句的类，按照要求生成诗句
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成诗句的最大长度，需小于等于model所允许的最大长度
    """

    # 类的初始化方法
    def __init__(self, model, tokenizer, max_length=512, poetry_len=0, poetry_type=""):
        self.model = model  # 模型实例
        self.tokenizer = tokenizer  # 分词器实例
        self.puncs = ['，', '。', '？', '；', ',', '.', '?', ';']  # 可能用到的标点符号集合
        self.max_length = max_length  # 生成诗句的最大长度
        self.poetry_type = poetry_type  # 诗的类型（如绝句、律诗）
        self.poetry_len = poetry_len  # 诗的长度
        self.type_choice = {"j": 4, "l": 8}  # 绝句和律诗的行数选择

    # 生成诗句的方法
    def generate(self, style='', head='', topk=2):
        """
        根据要求生成诗句
        style (str): 生成诗句的风格，写成诗句的形式，如“大漠孤烟直，长河落日圆。”
        head (str, list): 生成诗句的开头内容。若head为str格式，则head为诗句开始内容；
            若head为list格式，则head中每个元素为对应位置上诗句的开始内容（即藏头诗中的头）。
        topk (int): 从预测的topk中选取结果
        """

        head_index = 0  # 头部内容的索引初始化
        style_ids = self.tokenizer.encode(style)['input_ids']  # 将风格文本编码为ID
        style_ids = style_ids[:-1]  # 去掉编码后的结束标记

        # 判断头部内容是否为列表
        head_is_list = True if isinstance(head, list) else False
        # 如果头部内容是列表，编码列表中的第一个元素；如果不是，编码整个头部字符串
        if head_is_list:
            poetry_ids = self.tokenizer.encode(head[head_index])['input_ids']
        else:
            poetry_ids = self.tokenizer.encode(head)['input_ids']
        poetry_ids = poetry_ids[1:-1]  # 去掉编码后的开始和结束标记

        break_flag = False  # 初始化中断标志
        lines = 1  # 初始化诗句行数
        cnt = 0  # 初始化计数器
        # 循环生成诗句，直到达到最大长度或指定行数
        while len(style_ids) + len(poetry_ids) <= self.max_length and lines <= self.type_choice[self.poetry_type]:
            next_word = self._gen_next_word(style_ids + poetry_ids, topk)  # 生成下一个词
            # 跳过特殊符号
            if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                continue
            else:
                cnt += 1
            # 如果生成的是标点符号，重置计数器并增加行数
            if next_word in self.tokenizer.convert_tokens_to_ids(self.puncs):
                cnt = 0
                lines += 1
            # 如果不是标点符号且达到了指定长度，根据行数奇偶性添加逗号或句号
            elif next_word not in self.tokenizer.convert_tokens_to_ids(self.puncs) and cnt == self.poetry_len + 1:
                if lines % 2 == 0:
                    next_word = self.tokenizer.convert_tokens_to_ids('，')
                else:
                    next_word = self.tokenizer.convert_tokens_to_ids('。')
            new_ids = [next_word]  # 将选择的词添加到ID列表中
            # 如果是第一行且风格不为空，将风格ID和诗句ID合并
            if lines == 1 and style != '':
                poetry_ids = style_ids + poetry_ids
                style_ids = []
            # 如果生成的是结束标记，中断循环
            if next_word == self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                break
            poetry_ids += new_ids  # 更新诗句ID列表
            if break_flag:
                break
        # 清除特殊符号
        for ind, id in enumerate(poetry_ids, start=0):
            if id in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                poetry_ids.pop(ind)
        # 返回生成的诗句
        return ''.join(self.tokenizer.convert_ids_to_tokens(
            poetry_ids[:self.type_choice[self.poetry_type] * (self.poetry_len + 1)]))

    # 私有方法，用于生成下一个词的ID
    def _gen_next_word(self, known_ids, topk):
        type_token = [0] * len(known_ids)  # 初始化类型标记
        mask = [1] * len(known_ids)  # 初始化掩码
        sequence_length = len(known_ids)  # 已知ID的长度
        known_ids = paddle.to_tensor([known_ids], dtype='int64')  # 将已知ID转换为Paddle张量
        type_token = paddle.to_tensor([type_token], dtype='int64')  # 将类型标记转换为Paddle张量
        mask = paddle.to_tensor([mask], dtype='float32')  # 将掩码转换为Paddle张量
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)  # 使用模型进行前向传播
        words_prob = logits[0, -1, :].numpy()  # 获取最后一个词的概率分布
        words_to_be_choosen = words_prob.argsort()[::-1][:topk]  # 选择概率最高的topk个词
        probs_to_be_choosen = words_prob[words_to_be_choosen]  # 获取这些词的概率
        probs_to_be_choosen = probs_to_be_choosen / sum(probs_to_be_choosen)  # 对概率进行归一化
        word_choosen = np.random.choice(words_to_be_choosen, p=probs_to_be_choosen)  # 根据概率随机选择一个词
        return word_choosen  # 返回选择的词的ID
