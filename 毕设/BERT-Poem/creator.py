from tkinter import *
from model import PoetryBertModel
from gen import PoetryGen
from paddlenlp.transformers import BertTokenizer

import paddle
import re#导入必要的模块和库：导入了 tkinter 库用于构建 GUI，导入了模型和生成器相关的模块，导入了 BertTokenizer 用于分词。

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#加载模型和分词器：初始化了 BertTokenizer 和 PoetryBertModel，并设置了模型路径。
net = PoetryBertModel('bert-base-chinese', 128)
model_path = ['./checkpoint5j/checkpoint5j/final', './checkpoint5l/checkpoint5l/final', './checkpoint7j/checkpoint7j/final', './checkpoint7l/checkpoint7l/final']
def get_type():#用于获取用户选择的诗歌类型。
    # 獲取radiobutton的類型
    val = v.get()
    return val


def select_model(type):#用于根据用户选择的诗歌类型获取对应的模型路径。
    # 選擇模型路徑
    return model_path[type]


def poetry_show(poetry):#用于将生成的诗歌文本按照标点符号进行格式化输出。
        pattern = r"([，。；？])"
        text = re.sub(pattern, r'\1 ', poetry)
        for p in text.split():
            if p:
                print(p)


def gen_poem(style='', model_path='', poetry_len=0, poetry_type=''):#用于生成古诗。在函数中，加载了预训练模型，生成诗歌，并返回结果。
    

    # 载入已经训练好的模型
    
    model = paddle.Model(net)
    model.load(model_path)
    poetry_gen = PoetryGen(model, bert_tokenizer, poetry_len=poetry_len, poetry_type=poetry_type)
    
    # 随机生成一首诗
    poetry = poetry_gen.generate(style)
    del model
    return poetry
    # poetry_show(poetry)


def pretty_print_poem_5jue(word):#用于生成五言绝句，并通过 GUI 显示在界面上。
    # 輸出五言絕句
    poem_ = gen_poem(style=word, model_path=select_model(get_type()), poetry_len=5, poetry_type="j")
    poem_sentences = poem_.split('。')
    p = '\n'
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            s = str(s)+'。'+'\n'
            p = p + s
        label.config(text=p)


def pretty_print_poem_5lv(word):#用于生成五言律诗，并通过 GUI 显示在界面上。
    # 輸出五言律詩
    poem_ = gen_poem(style=word, model_path=select_model(get_type()), poetry_len=5, poetry_type="l")
    poem_sentences = poem_.split('。')
    p = '\n'
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            s = str(s)+'。'+'\n'
            p = p + s
        label.config(text=p)


def pretty_print_poem_7jue(word):
    # 輸出七言絕句
    poem_ = gen_poem(style=word, model_path=select_model(get_type()), poetry_len=7, poetry_type="j")
    poem_sentences = poem_.split('。')
    p = '\n'
    for s in poem_sentences:
        if s != '' and len(s) > 14:
            s = str(s)+'。'+'\n'
            p = p + s
        label.config(text=p)


def pretty_print_poem_7lv(word):
    # 輸出七言律詩
    poem_ = gen_poem(style=word, model_path=select_model(get_type()), poetry_len=7, poetry_type="l")
    poem_sentences = poem_.split('。')
    p = '\n'
    for s in poem_sentences:
        if s != '' and len(s) > 14:
            s = str(s)+'。'+'\n'
            p = p + s
        label.config(text=p)


def select_to_show():
    # 選擇輸出的詩文種類，根据用户选择的诗歌类型调用相应的生成函数，并在 GUI 中显示结果。
    if get_type() == 0:
        pretty_print_poem_5jue(et.get())
    elif get_type() == 1:
        pretty_print_poem_5lv(et.get())
    elif get_type() == 2:
        pretty_print_poem_7jue(et.get())
    else:
        pretty_print_poem_7lv(et.get())


if __name__ == '__main__':
    window = Tk()
    window.title("古詩文生成系統")
    frame1 = Frame(window)                                                     # 框架一 顯示詩文
    frame1.pack()
    label = Label(frame1, font=('方正楷体', 20), compound='center')
    label.pack()
    frame2 = Frame(window, background='Wheat')                                 # 框架二 操作功能
    frame2.pack()
    Label(frame2, text="請輸入詩的提示:", height=2, width=20, font=('方正楷体', 10),
          background='Wheat', borderwidth=0).pack(side=LEFT)
    et = Entry(frame2, width=20, background='BlanchedAlmond')                  # 定义一个输入框，用于输入詩的提示
    et.pack(side=LEFT)
    style = [('五絕', 0), ('五律', 1), ('七絕', 2), ('七律', 3)]                 # 定義一組radio button
    v = IntVar()
    for lan, var in style:
        Radiobutton(frame2, text=lan, value=var, command=get_type, variable=v,
                    font=('方正楷体', 10), background='Wheat', disabledforeground='Moccasin').pack(side=LEFT)
    Button(frame2, text="創作", command=select_to_show, height=2, width=10, font=('方正楷体', 10)
           , activebackground='Moccasin', background='BurlyWood',
           highlightbackground='BurlyWood', highlightcolor='BurlyWood').pack(side=RIGHT)  # 定义一个按钮
    window.mainloop()