# HKSL Financial Gloss LoRA Fine-Tuning Project

本项目旨在基于 Qwen 系列模型（Qwen2.5-7B / Qwen3-VL），通过 LoRA 微调技术，实现**金融粤语文本**到**香港手语（HKSL）Gloss 语法**的转换。

## 🖥️ 硬件与环境要求

- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **OS**: Ubuntu / Linux
- **CUDA**: 12.1+ (推荐)
- **Python**: 3.10+

## 🛠️ 1. 环境安装 (Installation)

首先建立独立的 Conda 环境，防止依赖冲突。

```bash
# 1. 创建并激活环境
conda create -n llama_env python=3.10
conda activate llama_env

# 2. 拉取 LLaMA-Factory 仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 3. 安装依赖 (包含 metrics 支持)
pip install -e .[metrics]

# 4. 安装 vLLM 用于后续推理服务
pip install vllm

📂 2. 数据准备 (Data Preparation)
2.1 数据格式

在 LLaMA-Factory/data 目录下创建 hksl_train.json。数据需符合 Alpaca 格式：
json

[
  {
    "instruction": "将以下金融粤语文本转换为符合香港手语(HKSL)语法的Gloss格式，仅输出结果。",
    "input": "恒生指数今日升咗三百点。",
    "output": "今日 恒生指数 升 三百 点"
  },
  {
    "instruction": "将以下金融粤语文本转换为符合香港手语(HKSL)语法的Gloss格式，仅输出结果。",
    "input": "如果你想开户口，要带身份证。",
    "output": "开户口 想 你？ 身份证 带 需要"
  }
]

2.2 注册数据集

编辑 LLaMA-Factory/data/dataset_info.json，在文件头部加入以下注册信息：
json

"hksl_data": {
  "file_name": "hksl_train.json"
},

🚀 3. 微调训练 (Fine-tuning)

使用 LLaMA-Factory 的 WebUI 进行可视化训练。
启动命令
bash

export CUDA_VISIBLE_DEVICES=0
llamafactory-cli webui

启动后访问：http://localhost:7860
⚙️ 关键参数配置 (针对 RTX 4090 优化)

请在 WebUI 中严格按照以下参数设置，以防止 OOM (显存溢出) 并保证效果：

    Model Name: 选择 Custom
    Model Path: /home/nvme_disk2/Miyeon_intern/lora/models/Qwen2.5-7B-Instruct (根据实际路径填写)
    Dataset: 选择 hksl_data
    LoRA Rank: 16
    LoRA Alpha: 32
    Quantization bit: 4 (⚠️ 必须选 4-bit，否则 24G 显存无法加载 7B/8B 模型进行训练)
    Batch Size: 1
    Gradient Accumulation: 8 (等效 Batch Size = 8)
    Learning Rate: 2e-4 (推荐)

点击 Start Training 开始炼丹。
⚡ 4. 模型推理与服务 (Inference & Serving)

训练完成后，使用 vllm 部署兼容 OpenAI API 的推理服务。
启动 API 服务

假设微调后的模型权重保存在 saves/Custom/lora/train_... 或已合并导出至 cut100 目录：
bash

python -m vllm.entrypoints.openai.api_server \
    --model /home/nvme_disk2/Miyeon_intern/lora/models/Qwen2.5-7B-Instruct-cut100 \
    --served-model-name cut100 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6

监控显卡状态

另开一个终端监控训练或推理时的显存占用：
bash

watch -n 1 nvidia-smi

📁 目录结构说明
text

/home/nvme_disk2/Miyeon_intern/lora/
├── data/                  # 原始训练数据
├── LLaMA-Factory/         # 训练框架核心代码
│   ├── data/              # 需要放入 hksl_train.json 的位置
│   └── ...
├── models/                # 基础模型存放处
│   ├── Qwen2.5-7B-Instruct
│   └── Qwen3-VL-8B-Instruct
└── saves/                 # 训练输出的 LoRA 权重
