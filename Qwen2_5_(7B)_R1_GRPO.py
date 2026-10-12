# -*- coding: utf-8 -*-
"""
Qwen2.5-7B GRPO强化学习训练R1模型
课程：LLM模型蒸馏与微调实操
功能：使用GRPO（Group Relative Policy Optimization）训练Qwen2.5-7B的推理能力
环境：AutoDL GPU实例，建议 A100/A800 40GB+
依赖：pip install unsloth vllm
"""

# ========================================
# Step 1: 模型加载（启用vLLM快速推理）
# ========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import unsloth
from unsloth import FastLanguageModel
import torch

max_seq_length = 768  # 4090 24G显存适配：从1024降低以避免OOM
lora_rank = 32  # 更大的rank让模型更智能，但训练更慢

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,  # 启用vLLM快速推理
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # 显存不足时可降低
)


# ========================================
# Step 2: LoRA配置
# ========================================

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ========================================
# Step 3: GSM8K数据准备
# ========================================

import re
from datasets import load_dataset, Dataset

# 系统提示词：定义推理输出格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """从XML格式文本中提取答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """从####标记文本中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    """加载GSM8K数据集"""
    data = load_dataset('/root/autodl-tmp/datasets/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


dataset = get_gsm8k_questions()


# ========================================
# Step 4: 奖励函数设计
# ========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励：检查答案是否正确（权重最高）"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励：检查答案是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励：完全符合XML格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励：基本符合XML格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """计算XML标签完整性得分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML标签计数奖励"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ========================================
# Step 5: GRPOTrainer训练
# ========================================

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,  # 每个问题生成4个候选答案（4090 24G适配，从6降低）
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()


# ========================================
# Step 6: 模型测试与保存
# ========================================

# 保存LoRA参数
model.save_lora("grpo_saved_lora")

# 测试模型推理
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048,
)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)


# ========================================
# 模型导出选项（按需取消注释）
# ========================================

# 保存为16bit浮点
# model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# 保存为4bit整数
# model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# 仅保存LoRA适配器
# model.save_pretrained_merged("model", tokenizer, save_method="lora")

# 保存为GGUF q4_k_m格式
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
