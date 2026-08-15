# 毕业设计：中国古典诗词 BERT 模型

本科毕业设计存档，主题为基于 BERT 的中国古典诗词表示与生成。项目在开发过程中经历了三个阶段/版本，此仓库按原始文件夹结构分别保留。仅上传代码、小型数据与文档；训练环境、模型 checkpoint 等大文件未上传，详见下方说明。

## 目录概览

### Poem-BERT-CCPoem-master

以清华大学人工智能研究院自然语言处理与社会人文计算研究中心发布的 **BERT-CCPoem**（`BERT_CCPoem_v1/`，在约92.6万首中国古典诗词、893万余句上预训练，用于生成诗句的向量表示）为基础，在其上编写的一批探索性脚本（`0.1.py` ~ `1.6.py`、`gen_.1.py`、`gen_vec_rep.py`），用于调用该预训练模型做诗句相似度检索等实验。

- `BERT_CCPoem_v1/` 仅保留 config、tokenizer、vocab 等小文件；预训练权重 `pytorch_model.bin`（154MB，超过 GitHub 单文件 100MB 限制）未上传，可从原项目获取。

### 毕设/1.0

更规范化的第一版训练流水线：`1prepare.py`（加载 PaddleNLP 内置 poetry 数据集并清洗）→ `2model.py`（基于 `bert-base-chinese` 搭建诗歌生成模型）→ `3train.py` → `4evaluate.py` → `5predict.py`，另有开发笔记 `ideas.md`（早期任务构思：问答系统 vs 文本生成方向的比较）。

### 毕设/BERT-Poem

基于 PaddlePaddle/PaddleNLP 的更完整实现，代码包括 `dataset.py`（诗歌数据集封装）、`model.py`、`train.py`、`gen.py`（生成）、`creator.py`。附带按诗体分类的训练语料 `data/`（绝句 `qijue_all.txt`、五言律诗 `wulv_all.txt`、五言绝句 `wujue_all.txt`，均为中国古典诗词，公有领域文本）。

- 七言律诗语料 `qilv_all.txt`（14.3MB）超过单文件 10MB 上限未上传；其余三个语料文件（均 <10MB）已上传。
- 4 组训练 checkpoint（`checkpoint5j/5l/7j/7l`，每组含 `final.pdopt` 900MB + `final.pdparams` 452MB，合计约 5.4GB）远超 GitHub 限制，未上传。

### 毕设（论文本身）

- `毕业设计-朱彦燕20098533I0110031.pdf`：最终提交的论文正文。
- `zyy.docx`：论文完整草稿（约550段落，含图表）。
- `bishe.docx`：早期提纲/草稿。
- `毕设中期- PPTMON.pptx`、`毕设期末.pptx`：中期与期末答辩 PPT。

## 未上传内容说明

- **虚拟环境与安装包**：`.venv`（3.4GB）、CUDA 工具包安装文件 `cudatoolkit-11.8.0-*.conda`（640MB）、`VC_redist.x64.exe`（14MB）——均为环境依赖，可按 `requirement.txt` 重新安装。
- **模型权重与训练 checkpoint**：`pytorch_model.bin`（154MB）、4 组 PaddlePaddle checkpoint（合计约5.4GB）。
- **压缩包本身**：`BERT-Poem.zip`（4.79GB）——已从中提取出代码与小型数据文件，压缩包内其余内容（checkpoint、IDE 配置、`__pycache__`）不再重复保留。
- **单个文件 >10MB** 的语料文件（`qilv_all.txt`）。
- **一份意外生成的"README.md"**：`毕设/1.0` 原目录下有一个 2.5MB 的 `README.md`，内容实际是 `.venv` 目录结构的 `tree` 打印输出（并非真正的项目说明），已跳过，改为本文件。
- IDE 项目配置目录 `.idea`（PyCharm）——保留了 `.vscode/1.3.py`（真正的源码文件，虽然位置在 `.vscode` 目录下）。
