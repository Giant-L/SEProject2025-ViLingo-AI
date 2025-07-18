#  SEProject2025-Vilingo-AI

> 这是软件工程实践2025课程，项目Vilingo的AI板块，主要负责Vilingo项目中的语音识别与打分部分中的ai模块，详细的功能流程如下所示：（本模块会使用pipeline作为主要思路，打包调用，简化流程）

**Pipeline: 视频与录音内容匹配度分析**

1. **输入 (Input):**
   - 一个视频文件 (`video.mp4`)
   - 一个用户录音文件 (`user_audio.wav`)
2. **阶段一：视频内容概括 (Video Summarization)**
   - **子步骤 1.1:** 从视频文件中提取音频。
   - **子步骤 1.2:** 使用 Whisper 将提取出的音频转换成文字（视频原始文稿）。
   - **子步骤 1.3:** 使用一个大型语言模型 (LLM) 将视频原始文稿进行概括，提炼出核心要点（视频内容摘要）。
3. **阶段二：用户录音处理 (User Audio Processing)**
   - 使用 Whisper 将用户录音文件转换成文字，并获取时间戳信息（用户转述文稿）。
4. **阶段三：内容比对与评分 (Comparison & Scoring)**
   - **核心任务:** 对比 **[视频内容摘要]** 和 **[用户转述文稿]**。
   - 计算两者在语义上的相似度。
   - 输出一个最终分数（例如 0-100分）。
5. **输出 (Output):**
   - 一个量化的分数（如 `85`）。
   - （可选）用户转述文稿的文本和时间戳信息。

![ai模块流程图](./assets/ai_tasks.png)



##  本模块的任务

把相关模型和代码都打包成一个docker镜像，方便前端同学获取docker镜像之后调用api拉取本模块的ai服务。

框架设计如下:

```
+----------------+      1. POST /analyze      +-----------------+      2. job_id      +-----------------+
|   Frontend     | ------------------------> | FastAPI Endpoint| ------------------> |   Frontend      |
| (Web/Mobile)   |      (Uploads Files)      | (API Server)    |      (立即返回)      | (显示处理中...) |
+----------------+                           +-------+---------+                     +-------+---------+
                                                     | 3. 触发后台任务                     | 6. GET /result/{job_id}
                                                     V                                     | (轮询查询结果)
                                            +-----------------+                            |
                                            | Background Task |                            |
                                            | (Celery / ARQ)  |                            |
                                            +-------+---------+                            |
                                                     | 4. 执行核心流水线                     |
                                                     V                                     |
+------------------------------------------------------------------------------------------+
|                                        Vilingo Core Pipeline                             |
|                                                                                          |
|  [文件存储] -> [S1: 视频处理] -> [S2: 用户音频处理] -> [S3: 对比评分] -> [S4: 结果存储]  |
|      |               |                   |                    |             |           |
|  (临时文件)  (Whisper->LLM)         (Whisper)        (SentenceTransformer) (数据库/缓存)  |
|                                                                                          |
+------------------------------------------------------------------------------------------+
```







##  环境配置

具体的环境安装包已经写入文件`reqirements.txt`中，配置环境的时候只需要：（推荐使用conda）

```bash
conda create -n asr python=3.9 -y
conda activate asr
pip install --upgrade pip
pip install -r requirements.txt
```



在安装系统依赖的时候，

* MacOS

```bash
brew install ffmpeg git
```



* Windows

1. 下载并解压FFmpeg静态包（https://www.gyan.dev/ffmpeg/builds/）
2. 解压后的bin目录添加到系统的环境变量PATH中。



#  工作流分析

在本项目中，我们需要完成一系列调用ai的工作，总结后共计需要三个ai，分别为

1. 文本概括ai - Qwen2 7B Instruct 通义千问2
2. 语义相似度分析ai - all-MiniLM-L6-v2 句向量对比ai
3. 音频文件识别ai - whisper-small



##  文本概括ai - microsoft/Phi-3-mini-4k-instruct

- **简介**: `Phi-3-mini` 是由微软公司推出的一款革命性的“小而精”的语言模型。它虽然体积小巧，但在高质量数据集的训练下，其性能在很多任务上可以媲美比它大一倍的7B（70亿）参数级别的模型。
- **资源需求**: 这是我们选择它的核心原因。`Phi-3-mini` 对内存（RAM）和显存（VRAM）的要求非常友好，非常适合在消费级硬件（如笔记本电脑）上进行本地部署和运行，尤其是在总内存有限（如8GB或16GB）的设备上。
- **为什么好**:
  1. **高性价比**: 它在保持强大英文理解和概括能力的同时，极大地降低了资源门槛，使我们的AI服务能够在更广泛的设备上流畅运行。
  2. **性能可靠**: 作为微软的明星开源模型，它的英文处理能力非常出色，完全能够胜任我们项目中对英文SRT字幕进行核心内容概括的任务。
  3. **速度优势**: 由于模型更小，它的推理（生成摘要）速度通常会比更大的模型更快，能为用户提供更及时的响应。

*(注：我们最初曾考虑使用 `Qwen2-7B` 等更大的模型，但为了确保应用能在8GB内存的Mac设备上稳定构建和运行，我们最终选择了资源占用更优化的 `Phi-3-mini` 作为文本概括模型。)*



##  文本对比ai - all-MiniLM-L6-v2 （句向量对比ai）

**简介:** 这些是专门用于将句子或段落转换成“向量”的小模型。第二个模型支持多种语言，包括中文。

**资源需求:** 极低！它们非常小（几百MB），主要在 **CPU** 上运行就足够快，几乎不占用显存。

**为什么好:** 速度极快、资源占用小、无需复杂Prompt、结果稳定。是进行文本相似度计算的业界标准方案。



##  Whisper模型

**whisper模型 **识别音频文件的时候会输出与评分标准相关的以下关键词：

| Whisper 字段                | 数据类型    | 字段含义                                            | 可用于的评分维度                     |
| --------------------------- | ----------- | --------------------------------------------------- | ------------------------------------ |
| `text`                      | `str`       | 完整转写文本（所有片段拼接）                        | Completeness / Grammar / Word Choice |
| `segments[].text`           | `str`       | 单句（片段）转写文本                                | Completeness / Fluency / Grammar     |
| `segments[].start`          | `float`     | 片段起始时间（秒）                                  | Fluency（停顿间隔分析）              |
| `segments[].end`            | `float`     | 片段结束时间（秒）                                  | Fluency（停顿间隔分析）              |
| `segments[].avg_logprob`    | `float`     | 模型对该片段文本的平均对数概率，越接近 0 置信度越高 | Fluency / Pronunciation Confidence   |
| `segments[].no_speech_prob` | `float`     | 片段为静音的概率（0–1）                             | Completeness（跳句 / 静默检测）      |
| `segments[].tokens`         | `List[int]` | 文本对应的 token ID 序列                            | 可选：Grammar / 词序分析             |
| `language`                  | `str`       | 自动检测语种（如 `"en"`）                           | 语言一致性校验                       |

本次使用的是whisper-small模型，模型大小为461MB，在一般的电脑上完全可以运行。



### **whisper模型的微调** 

主要是以欧美国家的英语母语者的对话为训练材料，然后在我们的项目中需要识别的对象是初学英语的中国人，因此有必要对原本的whisper模型做一些定向微调，以适应我们的项目需要。

**训练集**

| 数据集名称       | 内容简介                                                     | 规模 & 资源       | 语言/口音特点                        | 许可 & 访问                                                  |
| ---------------- | ------------------------------------------------------------ | ----------------- | ------------------------------------ | ------------------------------------------------------------ |
| **L2-ARCTIC**    | 英语非母语口音数据，包括 Mandarin（普通话）等6种母语的语音录制及标注 | ~24 小时，24 说者 | 包含 Mandarin = 中国人英语发音样本 † | CC BY-NC 4.0  [oai_citation:0‡HyperAI超神经](https://hyper.ai/en/datasets/8939?utm_source=chatgpt.com) |
| **Common Voice** | Mozilla 众包的多语种语音语料库，包含 Chinese (China) 方言语音样本 | 数百到上千小时    | 含不同方言的中国人说英语子集 ()      | CC0/Public Domain                                            |
| **OC16‑CE80**    | 中英混合语料库，以中文为主，包含嵌入少量英语单词或短语       | 80 小时语音       | 适合训练中式混合口音语料 ()          | 开源许可                                                     |
| **ASCEND**       | 香港口音—中英混合对话数据，可用于过渡语音识别任务            | ~10.6 小时        | 港式中英混杂场景 ()                  | 学术用途                                                     |

L2‑ARCTIC 是非英语母语者的语料库，这个数据集包含 Mandarin（普通话）说者录音，适合构建发音评测模型。  

使用的数据如下：

HKK(male mandarin): https://drive.google.com/file/d/13nGEWhGVELAiUoDQpSH3PcC4xoLFIwQ6/view?usp=sharing

MBMPS(female mandarin): https://drive.google.com/file/d/16CArT2LpGA1A7xJn_wGvzndsdeFTLVPs/view?usp=sharing

LXC(male mandarin): https://drive.google.com/file/d/1dY9BG-TTVB-14wz1f656EKBCYfv_IINp/view?usp=sharing

YKWK(female mandarin): https://drive.google.com/file/d/1Jq13epxqWmc-oJizvacjDTzVdIMHi5rG/view?usp=sharing

ZHAA(female cantonese): https://drive.google.com/file/d/1GrhaazNNU4iZvJJwshsxoiuPuBjMVBpA/view?usp=sharing



###  Demo

在配置好环境的时候，whisper第一次加载small模型时会自动从网上拉取预训练权重的进度条；461MB就是模型文件的大小。

whisper会把权重缓存在`~/.cache/whisper/`（或类似目录）下，之后再跑就不再下载了



第一次下载small模型并成功运行demo的截图如下：

![截屏2025-07-14 11.03.39](./assets/截屏2025-07-14 11.03.39.png)