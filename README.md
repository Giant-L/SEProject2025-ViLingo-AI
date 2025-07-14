#  SEProject2025-Vilingo-AI

> 这是软件工程实践2025课程，项目Vilingo的AI板块，主要负责Vilingo项目中的语音识别与打分部分中的ai模块，详细的功能流程如下所示：

 * 用户录音打分板块

   * 正确接受前端发送过来的用户录音

   * 使用whisper-small模型对录音文件进行转文本操作，获取到：

     * transcript:完整转写文本
     * Segments:每段话的开始/结束时间（时间戳列表）

   * 针对三大维度进行分析：

     * Fluency：

       * 语速
       * 停顿
       * 填充词

     * Grammer：

       * 开源语法检查LanguageTool
       * LLM校对

     * Completeness：

       * 关键词覆盖对比
       * 语义相似度对比

     * 打分与报告：

       归一化、加权、生成反馈

![ai模块流程图](./assets/ai_tasks.png)



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



* WIndows

1. 下载并解压FFmpeg静态包（https://www.gyan.dev/ffmpeg/builds/）
2. 解压后的bin目录添加到系统的环境变量PATH中。