# 毕业设计：中国古典诗词 BERT 模型

本科毕业设计存档，主题为基于 BERT 的中国古典诗词表示与生成。项目在开发过程中经历了三个阶段/版本，此仓库按原始文件夹结构分别保留代码。**这个仓库只是拼图的一块**——训练好的模型权重放在 Google Drive，部分训练数据是公开数据集，需要单独下载。下面的"项目拼图地图"列出了每一块在哪里、怎么拿到，跟着走可以在新设备上完整复现。

## 🧩 项目拼图地图

| 组成部分 | 在哪里 | 怎么拿到 |
|---|---|---|
| `Poem-BERT-CCPoem-master/` 探索脚本 | 本仓库 | `git clone` 本仓库即可 |
| BERT-CCPoem 预训练权重（162MB） | 清华官方发布 | 见下方"版本一"步骤，有直接下载命令 |
| `毕设/1.0/` 流水线代码 | 本仓库 | `git clone` 本仓库即可 |
| `毕设/1.0/` 训练数据 | PaddleNLP 内置，自动下载 | 首次运行 `1prepare.py` 时自动获取，无需手动准备 |
| `毕设/BERT-Poem/` 代码 | 本仓库 | `git clone` 本仓库即可 |
| 诗体语料（qijue/qilv/wujue/wulv_all.txt） | 网络公开数据集 | **⚠️ 具体来源待补充**——本地已删除（可重新下载），但尚未记录确切链接，需要回忆/找回后补充到这里 |
| 4 个训练好的 checkpoint（5j/5l/7j/7l，各约1.3GB，共约5.3GB） | Google Drive（私有） | **⚠️ 链接待补充**——上传后请把分享链接发给我，我会更新这里并直接替换掉这行占位 |
| 论文正文、草稿、答辩 PPT | 本仓库 `毕设/` | `git clone` 本仓库即可 |

## 目录概览与复现步骤

### 版本一：Poem-BERT-CCPoem-master（句向量检索）

以清华大学人工智能研究院自然语言处理与社会人文计算研究中心发布的 **BERT-CCPoem**（在约92.6万首中国古典诗词、893万余句上预训练）为基础，用一批探索性脚本（`0.1.py` ~ `1.6.py`、`gen_.1.py`、`gen_vec_rep.py`）调用该模型做诗句相似度检索实验。原始来源：[THUNLP-AIPoet/BERT-CCPoem](https://github.com/THUNLP-AIPoet/BERT-CCPoem)。

复现步骤：
1. `pip install torch>=1.2.0 transformers==4.3.3`
2. 下载并解压预训练权重到 `BERT_CCPoem_v1/` 目录下（与仓库里已有的 config/tokenizer/vocab 放在一起）：
   ```
   wget https://thunlp.oss-cn-qingdao.aliyuncs.com/BERT_CCPoem_v1.zip
   unzip BERT_CCPoem_v1.zip -d Poem-BERT-CCPoem-master/
   ```
3. 运行 `gen_vec_rep.py` 或其余探索脚本。

### 版本二：毕设/1.0（PaddleNLP 结构化流水线）

`1prepare.py`（加载 PaddleNLP 内置 poetry 数据集并清洗）→ `2model.py`（基于 `bert-base-chinese` 搭建生成模型）→ `3train.py` → `4evaluate.py` → `5predict.py`，另有开发笔记 `ideas.md`（早期任务构思：问答系统 vs 文本生成方向的比较）。

复现步骤：
1. `pip install paddlepaddle paddlenlp`（这个版本没有单独的 requirement.txt，依赖是从代码 import 推断的）
2. **注意**：`2model.py` 里直接引用了 `vocab` 变量但没有定义它——这几个脚本是按顺序在同一个交互式会话里执行的（写作时的思路草稿），不是能各自独立运行的脚本，直接 `python 2model.py` 会报 `NameError`。复现时建议把 5 个文件的内容依次粘贴进同一个 Jupyter/交互式 Python 会话。

### 版本三：毕设/BERT-Poem（分诗体训练，更完整的实现）

代码包括 `dataset.py`（诗歌数据集封装）、`model.py`、`train.py`、`gen.py`（生成逻辑）、`creator.py`（**tkinter 图形界面小程序**，可以选诗体、输入风格提示词，点按钮生成诗句）。四个 checkpoint 分别对应四种诗体：`5j`=五言绝句、`5l`=五言律诗、`7j`=七言绝句、`7l`=七言律诗。

复现步骤：
1. `pip install -r 毕设/BERT-Poem/requirement.txt`
2. 下载诗体语料（见上方地图，来源待补充），放进 `毕设/BERT-Poem/data/` 目录，文件名保持 `qijue_all.txt`/`qilv_all.txt`/`wujue_all.txt`/`wulv_all.txt`
3. **用已训练好的模型**：从 Google Drive 下载对应的 `checkpointXX.zip`，直接解压到 `毕设/BERT-Poem/` 目录下——压缩包内部路径已经是代码要求的双层嵌套格式（如 `checkpoint5j/checkpoint5j/final.pdopt`），解压后 `test.py`/`creator.py` 里的 `model.load('./checkpoint5j/checkpoint5j/final')` 可以直接读到。
4. **从头训练**：`train.py` 目前是写死加载 `qilv_all.txt` 并存到 `./checkpoint7l` 的，训练其他 3 个诗体需要手动改这两处（数据文件名 + `save_dir`），跑 4 次。
5. 生成诗句：命令行用 `python test.py`，图形界面用 `python creator.py`。

### 毕设（论文本身）

- `毕业设计-朱彦燕20098533I0110031.pdf`：最终提交的论文正文。
- `zyy.docx`：论文完整草稿（约550段落，含图表）。
- `bishe.docx`：早期提纲/草稿。
- `毕设中期- PPTMON.pptx`、`毕设期末.pptx`：中期与期末答辩 PPT。

## 未上传到 GitHub 的内容

- **虚拟环境与安装包**：`.venv`、CUDA 工具包安装文件、`VC_redist.x64.exe`——均为环境依赖，按上面复现步骤里的 `pip install` 重新安装即可，本地也已删除这些以腾出磁盘空间。
- **模型权重与训练 checkpoint**：`pytorch_model.bin`（154MB，见版本一步骤 2 下载）、4 组 PaddlePaddle checkpoint（合计约5.3GB，放在 Google Drive，见上方地图）。
- **诗体语料**（qijue/qilv/wujue/wulv_all.txt，网络公开数据集）：本地也已删除，具体来源待补充到上方地图里。
- **压缩包本身**：原始 `BERT-Poem.zip`（4.79GB）——已从中提取出代码（`BERT-Poem_code.zip`）与 4 个 checkpoint（分别打包上传 Google Drive），压缩包内其余内容（IDE 配置、`__pycache__`）不再保留。
- **一份意外生成的"README.md"**：`毕设/1.0` 原目录下曾有一个 2.5MB 的文件，内容实际是 `.venv` 目录结构的 `tree` 打印输出（并非真正的项目说明），已跳过。
- IDE 项目配置目录 `.idea`——保留了 `.vscode/1.3.py`（真正的源码文件，虽然位置在 `.vscode` 目录下）。

## 本地磁盘整理记录（E:\learn\毕设）

为腾出磁盘空间，本地对该项目做了以下整理（GitHub 仓库内容不受影响，代码保持原样可浏览）：
- `毕设/1.0/` 整个文件夹（含 3.4GB 的 `.venv`、640MB CUDA 安装包等）已删除，代码压缩保存为 `毕设/1.0_code.zip`。
- `毕设/BERT-Poem.zip`（4.79GB）里的代码已提取压缩为 `毕设/BERT-Poem_code.zip`；4 个 checkpoint 已分别打包为 `毕设/checkpoints_for_upload_zips/checkpointXX.zip`，待上传 Google Drive 后即可删除原始大压缩包。
- `E:\Poem-BERT-CCPoem-master` 未改动，保持原样。
