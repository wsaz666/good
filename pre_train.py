import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ================= 1. 配置路径 =================
tokenizer_path = "./verilog_tokenizer_32k"
dataset_path = "D:/code/code generation/data/train_final_eos.csv"
output_dir = "./verilog_gpt2_stage1_pretrain"


# ================= 预处理函数定义 =================
# 注意：函数定义必须放在主程序外部，以便多进程调用
def get_preprocess_function(tokenizer):
    def preprocess_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=1024,
        )

    return preprocess_function


# ================= 主程序入口 =================
def main():
    # 显卡配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 2. 加载分词器 =================
    print(f"正在加载专用分词器: {tokenizer_path} ...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        tokenizer.pad_token = "<|padding|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = tokenizer.eos_token
    except Exception as e:
        print(f"加载分词器失败！请先运行 train_new_setup.py 生成分词器。\n错误: {e}")
        return

    # ================= 3. 初始化模型 =================
    print("正在初始化 GPT-2 (12层/768维) ...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        gradient_checkpointing=True
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    print(f"模型参数量: {model.num_parameters() / 1e6:.2f} M")

    # ================= 4. 数据加载与处理 =================
    print(f"正在加载数据: {dataset_path} ...")
    dataset = load_dataset("csv", data_files=dataset_path)["train"]

    print("正在对数据进行 Tokenize (这可能需要几分钟)...")

    # 获取绑定了 tokenizer 的处理函数
    process_func = get_preprocess_function(tokenizer)

    # 【关键修改】多进程处理必须在 main 块保护下运行
    tokenized_dataset = dataset.map(
        process_func,
        batched=True,
        num_proc=4,  # Windows 下这里会触发 spawn
        remove_columns=["content"]
    )

    # 划分验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=5000, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # ================= 5. 训练参数配置 =================
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,             #把数据完整看5遍
        per_device_train_batch_size=4,  #显卡一次读4条数据
        gradient_accumulation_steps=8,  #累计8次更新参数
        learning_rate=2e-4,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ================= 6. 开始训练 =================
    print("🚀 开始阶段一预训练 (Pre-training)...")
    trainer.train()
    print("🎉 阶段一训练完成！模型已保存至:", output_dir)


if __name__ == "__main__":
    # Windows 必须加这一行来防止多进程递归报错
    main()