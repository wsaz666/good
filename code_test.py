import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextStreamer

# ================= 配置 =================
# 1. 分词器路径
tokenizer_path = "./verilog_tokenizer_32k"

# 2. 模型路径 (指向你微调结束后的文件夹)
# 注意：Trainer 可能会保存为 checkpoint-xxx，也可能保存了最终版
# 请去 ./verilog_gpt2_stage2_sft 文件夹里看一眼，如果有 checkpoint 文件夹，就填那个
# 如果有 pytorch_model.bin 直接在根目录，就填 ./verilog_gpt2_stage2_sft
model_path = "./verilog_gpt2_stage2_sft/checkpoint-666"
# 如果你发现文件夹里只有 checkpoint-xxxx，请把上面改成 "./verilog_gpt2_stage2_sft/checkpoint-xxxx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 加载模型 =================
print(f"正在加载分词器: {tokenizer_path} ...")
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = "<|padding|>"
tokenizer.eos_token = "<|endoftext|>"

print(f"正在加载微调后的模型: {model_path} ...")
try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！可以开始对话了。")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("请检查 model_path 是否指向了包含 config.json 和 pytorch_model.bin 的具体文件夹！")
    exit()


# ================= 对话循环 =================
def chat():
    print("\n" + "=" * 50)
    print("Verilog-Coder 交互模式 (输入 'quit' 退出)")
    print("提示：试着输入 'Create a module named counter with clk, rst, out.'")
    print("=" * 50)

    # 创建流式输出器 (像 ChatGPT 一样打字)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    while True:
        # 1. 获取用户输入
        instruction = input("\n[User]: ").strip()
        if instruction.lower() in ["quit", "exit"]:
            break
        if not instruction:
            continue

        # 2. 构造 SFT 格式的 Prompt
        # 必须和训练时保持一致：<instruction>: ... \n<output>:
        prompt = f"<instruction>: {instruction}\n<output>:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("\n[AI]: ", end="")

        # 3. 生成代码
        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,  # 允许它写长一点

                # === 推理参数调优 (SFT 模型不需要太激进的惩罚了) ===
                do_sample=True,
                temperature=0.6,  # 0.6 比较稳，适合写代码
                top_p=0.95,
                repetition_penalty=1.1,  # 轻微惩罚即可，因为它已经学会不复读了

                # 停止符 (非常重要)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,

                streamer=streamer  # 实时输出
            )
        print("-" * 30)


if __name__ == "__main__":
    chat()