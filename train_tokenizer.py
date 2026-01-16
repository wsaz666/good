import os
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

# ================= 配置 =================
corpus_file = "D:/code/code generation/data/verilog_corpus.txt"
tokenizer_save_dir = "./verilog_tokenizer_32k"

# 词表大小：32k (相比 Qwen 的 150k，减少了 80% 的 Embedding 参数)
# 这样我们可以把参数量用在增加层数上
VOCAB_SIZE = 32000


def train_tokenizer():
    print(f"正在训练 Verilog 专用分词器 (Vocab={VOCAB_SIZE})...")

    # 使用 BPE 算法
    tokenizer = ByteLevelBPETokenizer()

    # 训练
    tokenizer.train(
        files=[corpus_file],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,  # 出现少于2次的词丢弃
        special_tokens=["<|endoftext|>", "<|padding|>"]
    )

    # 保存
    if not os.path.exists(tokenizer_save_dir):
        os.makedirs(tokenizer_save_dir)
    tokenizer.save_model(tokenizer_save_dir)

    print("分词器训练完成！")

    # 将其转换为 Transformers 格式并验证
    from transformers import GPT2TokenizerFast
    # 手动加载，确保 EOS token 正确
    fast_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_save_dir)
    fast_tokenizer.pad_token = "<|padding|>"
    fast_tokenizer.eos_token = "<|endoftext|>"
    fast_tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"Transformers 格式分词器已保存至: {tokenizer_save_dir}")

if __name__ == "__main__":
    # 1. 先训练分词器
    if os.path.exists(corpus_file):
        train_tokenizer()
    else:
        print(f"错误：找不到语料文件 {corpus_file}，请先运行清洗脚本！")
        exit()