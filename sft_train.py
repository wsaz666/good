import os
import torch
import numpy as np
import sacrebleu
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ================= 1. 全局配置与资源加载 =================
# 为了防止 Windows 多进程报错，我们将 tokenizer 在全局加载
# 这样子进程 import 脚本时也能直接获取到 tokenizer 对象
tokenizer_path = "./verilog_tokenizer_32k"
model_stage1_path = "./verilog_gpt2_stage1_pretrain/checkpoint-8595"
dataset_path = "D:/code/code generation/data/train_platinum_distilled.csv"
output_dir = "./verilog_gpt2_stage2_sft"

print(f"正在加载分词器: {tokenizer_path} ...")
try:
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.eos_token = "<|endoftext|>"
except Exception as e:
    print(f"⚠️ 全局分词器加载警告: {e}")
    tokenizer = None

# ================= 2. 核心函数 (定义在全局) =================

def preprocess_function(examples, tokenizer):
    # 注意：这里接收通过 fn_kwargs 传入的 tokenizer
    inputs = examples["instruction"]
    outputs = examples["output"]

    new_texts = []
    for inst, out in zip(inputs, outputs):
        # SFT 格式: <instruction>: ... \n <output>: ... <|endoftext|>
        text = f"<instruction>: {inst}\n<output>: {out}{tokenizer.eos_token}"
        new_texts.append(text)

    return tokenizer(
        new_texts,
        truncation=True,
        max_length=1024,
    )


def preprocess_logits_for_metrics(logits, labels):
    """
    显存优化：在 GPU 上即时将 Logits 降维为 Token ID。
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # ================= 修复开始 =================
    # 1. 如果 preds 是元组（有些模型会输出 tuple），取第一个元素
    if isinstance(preds, tuple):
        preds = preds[0]

    # 2. 【关键修复】将预测结果中的 -100 替换为 pad_token_id
    # Trainer 会在 batch 对齐时自动填入 -100，这会导致 tokenizer 报错！
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # 3. 将标签中的 -100 也替换为 pad_token_id (为了解码不报错)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # ================= 修复结束 =================

    # 解码
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 去除空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = {}

    # === BLEU 计算 (使用原生 sacrebleu) ===
    import sacrebleu
    try:
        # 简单打印一条看看，确保解码正常
        # print(f"[Debug] Pred: {decoded_preds[0][:50]}...")

        # sacrebleu.corpus_bleu 需要 references 是 [[ref1_a, ref2_a], [ref1_b, ref2_b]]
        # 我们只有单参考，所以需要转置一下： [decoded_labels_clean] -> [[label1, label2, ...]]
        # 注意：decoded_labels 目前是 [['label1'], ['label2']]
        # 我们需要把它变成 [['label1', 'label2', ...]] 的形式给 corpus_bleu

        # 修正 references 的格式
        refs = [[l[0] for l in decoded_labels]]

        bleu = sacrebleu.corpus_bleu(decoded_preds, refs)
        result["bleu"] = bleu.score
    except Exception as e:
        print(f"⚠️ BLEU 计算报错: {e}")
        result["bleu"] = 0.0
    # ======================================

    # 计算 Token Accuracy
    mask = labels != tokenizer.pad_token_id
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0
    result["token_accuracy"] = accuracy

    return result


# ================= 3. 主程序入口 =================
if __name__ == "__main__":
    # 显卡检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 二次检查 Tokenizer
    if tokenizer is None:
        print("❌ 分词器未正确加载，程序退出。")
        exit()

    # --- 加载模型 ---
    print(f"正在加载阶段一模型: {model_stage1_path} ...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_stage1_path)
        model.to(device)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    # --- 数据处理 ---
    print(f"正在加载数据: {dataset_path} ...")
    dataset = load_dataset("csv", data_files=dataset_path)["train"]

    print("正在构建对话数据...")
    # Windows 下使用 map 且 num_proc > 1 时，
    # 调用的函数 preprocess_function 必须是全局定义的
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}  # 【关键】显式传递 tokenizer
    )

    split = tokenized_dataset.train_test_split(test_size=2000, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # --- 训练配置 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 初始化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    # --- 开始训练 ---
    print("🚀 开始阶段二指令微调 (SFT)...")
    trainer.train()
    print("🎉 完成！")