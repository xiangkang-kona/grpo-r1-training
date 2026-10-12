# GRPO 强化学习训练 Qwen2.5-7B 推理模型（R1 式 `<reasoning>/<answer>` 输出）

使用 GRPO（Group Relative Policy Optimization）算法对 Qwen2.5-7B-Instruct 做强化学习，让模型在无人工标注的情况下自主学习数学推理能力，并自发形成 `<reasoning>...</reasoning><answer>...</answer>` 的结构化推理格式——即 DeepSeek-R1 报告中"推理格式涌现"的复现实验。

## 原理简介

GRPO 与 PPO 的核心区别：**去掉 Critic 价值网络，用同一问题的多个候选回答的组内相对奖励作为基线**。对每个问题采样 N 个回答，按奖励函数打分后计算组内优势，胜者被强化、败者被抑制。训练奖励完全由规则函数给出（答案对不对、格式对不对），**全程不需要人工标注数据**。

本项目对每个问题采样 4 个候选回答（`num_generations=4`），设计 5 个奖励函数叠加驱动学习：

| 奖励函数 | 分值 | 作用 |
|---|---|---|
| `correctness_reward_func` | 2.0 | 答案与 GSM8K 标准答案一致（权重最高） |
| `int_reward_func` | 0.5 | 答案是纯数字（GSM8K 答案均为整数） |
| `strict_format_reward_func` | 0.5 | 完全符合 `<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n` 格式 |
| `soft_format_reward_func` | 0.5 | 基本符合标签结构 |
| `xmlcount_reward_func` | 0~0.5 | 逐标签计分并惩罚多余内容 |

系统提示词要求模型按 XML 格式作答，但**是否真的遵守完全靠模型在训练中自己探索**——这正是 GRPO 的意义。

## 实测结果（AutoDL RTX 4090 24G 单卡）

| 指标 | 数值 |
|---|---|
| 训练步数 | 250 steps（GSM8K train 7473 条，每步 1 题采样 4 答） |
| 训练耗时 | 约 31 分钟 |
| 显存峰值 | 约 16 GB（4bit 量化 + vLLM 推理引擎共存） |
| 格式奖励变化 | xmlcount 0.031 → 0.125~0.156（第 1 步 → 第 100+ 步） |
| 输出长度变化 | 平均 179 tokens → 122 tokens（学会更简洁地收敛到答案） |
| 输出格式 | 从裸文本推理逐步自发形成标准 XML 标签格式 |

训练后用 vLLM 加载 LoRA 实测（问题：Calculate pi），输出为标准格式：

```
<reasoning>
Calculating the exact value of pi is impossible due to its nature as an
irrational number... One such method is the Leibniz formula: π = 4 × (1 - 1/3
+ 1/5 - 1/7 + 1/9 - ...). ...
...</reasoning>
<answer>
The value of pi, to a common approximation, is 3.14159.
...</answer>
```

## 环境要求

```
torch==2.5.1
transformers==4.51.3
trl==0.15.2
unsloth==2025.3.19
unsloth_zoo==2025.3.17
vllm==0.7.3
datasets==3.1.0
peft==0.15.0
```

- **仅支持 Linux**（unsloth 的 Triton 加速内核不支持原生 Windows，建议 AutoDL 等 GPU 服务器）
- 显存要求：4090 24G 可跑（已调参适配）；A100/A800 40G+ 更宽裕
- 脚本内置 HF 镜像（hf-mirror.com）配置，国内服务器可直接下载模型与数据

## 运行前准备

### 0. 本仓库自带 GSM8K 数据集

`gsm8k/` 目录已包含 main 与 socratic 两个 config 的 parquet 数据及 configs 元数据，克隆后直接使用仓库相对路径即可：

```python
data = load_dataset('./gsm8k', 'main')[split]
```

若将数据集放到其他位置，同步修改脚本中 `get_gsm8k_questions()` 里的路径。

### 1. 模型下载（modelscope，约 15GB）

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')
```

### 2. GSM8K 数据集

`load_dataset('/root/autodl-tmp/datasets/gsm8k', 'main')` 按本地目录加载，目录需保持 HF 仓库结构（`README.md` 携带 configs 元数据 + `main/*.parquet`），否则按 config 名加载会失败。

### 3. 路径核对

确认脚本中两处路径与实际一致：

- 模型：`/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct`
- 数据：`/root/autodl-tmp/datasets/gsm8k`

## 运行

```bash
python Qwen2_5_(7B)_R1_GRPO.py
```

流程：4bit 加载基座 → 挂 LoRA（r=32，全部注意力+MLP 层）→ 启动 vLLM 快速推理引擎（`fast_inference=True`，显存占比 0.6）→ GRPO 训练 250 步 → 保存 LoRA（`grpo_saved_lora/`）→ vLLM 加载 LoRA 推理验证。

## 4090 24G 适配说明（相对原版调参）

| 参数 | 原版 | 适配值 | 说明 |
|---|---|---|---|
| `num_generations` | 6 | 4 | 每题采样数，直接决定训练时显存峰值 |
| `max_seq_length` | 1024 | 768 | 限制单条序列长度 |
| `gpu_memory_utilization` | 0.6 | 0.6（保持） | vLLM 显存占比，实测 16G 峰值无 OOM |

显存仍不足时的进一步退让顺序：`num_generations` → 3；`max_completion_length` 降到 384；`gpu_memory_utilization` 降到 0.5。

## 训练产物与使用

- `grpo_saved_lora/`：GRPO 训练得到的 LoRA 适配器
- `outputs/checkpoint-*`：训练检查点（`save_steps=250`），可断点续训

加载训练后的模型推理：

```python
from unsloth import FastLanguageModel
from vllm import SamplingParams

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=768, load_in_4bit=True,
    fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.6,
)

text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "你的问题"},
], tokenize=False, add_generation_prompt=True)

out = model.fast_generate(text, sampling_params=SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048),
                          lora_request=model.load_lora("grpo_saved_lora"))
print(out[0].outputs[0].text)
```

末尾注释掉的导出代码支持 merged 16bit / merged 4bit / 纯 LoRA / GGUF（q4_k_m，可进 Ollama）四种导出方式。

## 训练观察要点（日志解读）

- 每步日志中的 `rewards/xmlcount_reward_func` 等字段反映格式学习的进程：从接近 0 涨到 0.125 以上说明模型开始稳定输出标签结构
- `completion_length` 下降说明模型学会"想清楚就收尾"，不再空转凑长度
- `correctness_reward_func` 长期为 0 属正常——GSM8K 数学题 250 步内正确率提升有限，格式能力才是本实验的主要涌现目标；追求正确率需加大步数或换更强的奖励信号
- 训练结束时的 NCCL `process group has NOT been destroyed` 警告为退出清理提示，无实际影响

## 常见问题

- **vLLM 引擎初始化卡住或崩**：显存被其他进程占用，`nvidia-smi` 确认 GPU 空闲后重跑
- **加载本地 GSM8K 报 config 错误**：数据目录缺 `README.md`（configs 元数据靠它解析）
- **训练极慢**：确认 vLLM 生效（日志出现 vLLM 引擎 banner）；未生效时每步回退到 transformers 逐条生成，慢一个数量级
- **与 SFT 的关系**：SFT 学"标准答案长什么样"，GRPO 学"哪种回答得分高"；本脚本演示的规则奖励 GRPO 与老师模型标注 + SFT 蒸馏（见同目录 `Qwen2_5_(7B)_Alpaca.py`）是两种互补的能力获取方式
