# 毕业设计：基于预训练 BERT 的中国古典诗词生成

澳门科技大学本科毕业设计（2023.09–2024.05，个人项目）。主题为基于 BERT 的中国古典诗词表示与生成；开发过程中经历三个版本，本仓库按原始目录结构保留各版本代码、论文与答辩材料。

*Undergraduate thesis (Macau University of Science and Technology, Sep 2023 – May 2024, individual): classical Chinese poetry generation with pre-trained BERT. The repository keeps the code of the three development stages, the thesis and the defence slides.*

## 项目概述

最终系统（版本三，`毕设/BERT-Poem/`）基于 PaddleNLP 的开源诗歌生成示例代码（`PoemData` / `PoetryBertModel` / `PoetryBertModelLossCriterion` 的结构）改造：以 `bert-base-chinese` 为底座，使用下三角（lower-triangular）注意力掩码实现从左到右的生成；语料按 6:2:2 划分；按诗体分别训练 4 个模型（五言绝句、五言律诗、七言绝句、七言律诗）；训练使用 AdamW、学习率 1e-4、训练 batch 128 / 验证 batch 32、10 个 epoch、序列长度 128，训练时监控 Perplexity；生成时逐字 top-k 采样（`gen.py`，默认 k=2）；配有 Tkinter 图形界面（`creator.py`），可选诗体并输入风格提示词生成。

论文比较了 BERT-CCPoem 与 GPT-2 两种方法的生成效果，答辩 PPT 另介绍了最终实现方案。比较以定性分析为主，论文与答辩中没有定量评测指标表。直接使用 BERT-CCPoem / `bert-base-chinese` 的第一次生成尝试效果不佳（答辩 PPT 中的 "A failed attempt"），之后转向可训练的 Paddle 方案。

## 目录与材料

| 内容 | 位置 |
|---|---|
| 论文（英文，35 页）：背景综述、BERT-CCPoem 与 GPT-2 的定性对比 | `毕设/毕业设计-朱彦燕….pdf` |
| 答辩 PPT（中期、期末，期末为 2024-05-08） | `毕设/毕设中期- PPTMON.pptx`、`毕设/毕设期末.pptx` |
| 论文草稿 | `毕设/zyy.docx`、`毕设/bishe.docx` |
| 版本三：最终系统（4 个诗体模型 + Tkinter 界面） | `毕设/BERT-Poem/`（`dataset.py`、`model.py`、`train.py`、`gen.py`、`creator.py`、`test.py`） |
| 版本二：PaddleNLP 结构化流水线 | `毕设/1.0/`（`1prepare.py` → `5predict.py`，开发笔记 `ideas.md`） |
| 版本一：BERT-CCPoem 句向量检索探索脚本 | `Poem-BERT-CCPoem-master/` |

## 复现说明

**未包含在仓库中的内容：** 4 个训练好的 checkpoint（各约 1.3 GB，分别对应五绝/五律/七绝/七律）、诗体语料（`qijue_all.txt`、`qilv_all.txt`、`wujue_all.txt`、`wulv_all.txt`，网络公开数据集）、BERT-CCPoem 预训练权重（约 160 MB，清华发布）、虚拟环境与安装包。

**版本三（BERT-Poem）：**
1. `pip install -r 毕设/BERT-Poem/requirement.txt`
2. 将四个诗体语料放入 `毕设/BERT-Poem/data/`，文件名保持 `qijue_all.txt`/`qilv_all.txt`/`wujue_all.txt`/`wulv_all.txt`。
3. 训练：`train.py` 当前写死加载 `qilv_all.txt` 并保存到 `./checkpoint7l`；训练其他三个诗体需要修改这两处（数据文件名与 `save_dir`）。
4. 生成：命令行 `python test.py`，图形界面 `python creator.py`。模型加载路径为双层嵌套目录，例如 `./checkpoint5j/checkpoint5j/final`。

**版本二（1.0）：** `pip install paddlepaddle paddlenlp`（没有单独的 requirement 文件）。五个脚本是在同一个交互式会话中依次执行的草稿（例如 `2model.py` 直接使用前一个脚本里的变量 `vocab`，自身没有定义），不能各自独立运行。

**版本一：** `pip install torch>=1.2.0 transformers==4.3.3`，下载 [BERT-CCPoem](https://github.com/THUNLP-AIPoet/BERT-CCPoem) 预训练权重并解压到 `Poem-BERT-CCPoem-master/BERT_CCPoem_v1/`，运行 `gen_vec_rep.py` 等脚本。
