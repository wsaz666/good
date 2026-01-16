import json
import re
import os
from tqdm import tqdm

# ================= 配置 =================
input_file = "D:/code/code generation/data/Second_Cleaning_Dataset.json"
# 输出两个文件：
# 1. 用于训练 Tokenizer 的纯文本文件
tokenizer_corpus_file = "D:/code/code generation/data/verilog_corpus.txt"
# 2. 用于训练模型的 CSV (包含 EOS)
final_train_file = "D:/code/code generation/data/train_final_eos.csv"

TARGET_COUNT = 60000  # 提取 6 万条

# ================= 黑名单 (拒绝网表和工艺库) =================
BLACKLIST = [
    "sky130", "tsmc", "smic", "umc", "gf180",
    "dffe", "dff", "lut4", "lut6", "ibuf", "obuf", "bufg",
    "vivado", "quartus", "qsys", "nios",
    "timescale", "specify", "endspecify", "primitive"
]


def clean_code(code):
    # 1. 基础过滤
    if len(code) < 50 or len(code) > 10000:
        return None

    code_lower = code.lower()
    for term in BLACKLIST:
        if term in code_lower:
            return None

    # 2. 去除网表特征 (大量逗号)
    if code.count(",") > 50 and code.count(";") < 5:
        return None

    # 3. 去除版权声明 (Copyright 头)
    # 匹配以 // 开头，包含 Copyright 的行，直到空行
    code = re.sub(r"//\s*Copyright[\s\S]*?\n\s*\n", "", code, flags=re.IGNORECASE)

    # 4. 去除多余空行和制表符
    # 将 Tab 替换为 4 个空格
    code = code.replace("\t", "    ")
    # 将连续 3 个以上的换行替换为 2 个 (保留段落感但不过分)
    code = re.sub(r"\n{3,}", "\n\n", code)

    # 5. 去除非 ASCII 字符 (乱码)
    # 只保留 ASCII 可打印字符和换行
    code = re.sub(r'[^\x00-\x7F]+', '', code)

    return code.strip()


def process_data():
    print("开始清洗并构建语料库...")

    count = 0
    import csv

    # 打开文件准备写入
    f_txt = open(tokenizer_corpus_file, "w", encoding="utf-8")
    f_csv = open(final_train_file, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    writer.writerow(["content"])  # 只需要 content 列，自回归训练不需要 instruction/output 分离

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if count >= TARGET_COUNT:
                break
            try:
                data = json.loads(line)
                raw_code = data.get("text", "")

                # 清洗
                cleaned_code = clean_code(raw_code)
                if not cleaned_code:
                    continue

                # === 关键步骤：添加 EOS Token ===
                # 这告诉模型：“这个模块结束了，后面是下一个文件，不要强行找逻辑联系”
                # 我们使用标准的 GPT-2 EOS 符号
                final_text = cleaned_code + "\n<|endoftext|>\n"

                # 写入 Tokenizer 训练语料 (纯文本)
                f_txt.write(final_text)

                # 写入模型训练数据 (CSV)
                writer.writerow([final_text])

                count += 1
                if count % 5000 == 0:
                    print(f"已处理 {count} 条...")

            except Exception:
                continue

    f_txt.close()
    f_csv.close()
    print(f"完成！\nTokenizer 语料: {tokenizer_corpus_file}\n训练数据: {final_train_file}")


if __name__ == "__main__":
    process_data()