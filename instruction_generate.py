import asyncio
import os
import csv
from openai import AsyncOpenAI
from tqdm import tqdm  # 使用最标准的 tqdm

# ================= 配置 =================
API_KEY = "sk-ppflhpaeeazyczvvnlirhmywjvwmlsufxycrppgbpqfdgxcy"  # 【请填入你的 Key】
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 推荐用这个免费且强力的代码模型

# 输入输出文件路径
INPUT_FILE = "D:/code/code generation/data/train_final_eos.csv"
OUTPUT_FILE = "D:/code/code generation/data/train_platinum_distilled.csv"

CONCURRENCY_LIMIT = 5  # 并发数
TARGET_COUNT = 60000

# ================= 系统 Prompt =================
SYSTEM_PROMPT = """
你是一位资深的芯片设计专家。我会给你一段 Verilog 代码。
请你为这段代码编写一个详细的“指令 (Instruction)”。
要求：
1. 提取模块名称。
2. 清晰列出所有的输入(Input)和输出(Output)端口。
3. 简要描述模块的功能逻辑。
4. 只输出指令内容，不要输出其他废话。
5. 使用英文撰写。
"""

aclient = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)


async def generate_instruction(code):
    """
    核心修改点：
    不仅返回生成的 instruction，还把输入的 code 原样返回。
    这样在异步乱序完成时，我们可以把它们重新对应起来。
    """
    async with sem:
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Code:\n{code}"}
                ],
                temperature=0.2,
                max_tokens=200
            )
            # 返回 (结果, 原始代码)
            return response.choices[0].message.content.strip(), code
        except Exception as e:
            # 打印错误方便调试 (可选)
            # print(f"Error: {e}")
            return None, code


async def main():
    # ================= 1. 读取 CSV 并清洗 =================
    print(f"正在读取文件: {INPUT_FILE} ...")
    data_list = []

    # 防止大字段报错
    csv.field_size_limit(1000000)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader, None)  # 跳过表头
        except StopIteration:
            pass  # 空文件

        for row in reader:
            if len(data_list) >= TARGET_COUNT: break
            if not row: continue

            raw_text = row[0]

            # 清洗 EOS 标记
            clean_code = raw_text.replace("\n<|endoftext|>\n", "").replace("<|endoftext|>", "").strip()

            if len(clean_code) > 20:
                data_list.append(clean_code)

    print(f"有效代码条数: {len(data_list)}")

    # ================= 2. 检查断点续传 =================
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                processed_count = sum(1 for _ in reader) - 1
        except:
            processed_count = 0

    if processed_count < 0: processed_count = 0
    print(f"跳过已处理的前 {processed_count} 条...")

    # ================= 3. 异步处理循环 =================
    mode = "a" if os.path.exists(OUTPUT_FILE) and processed_count > 0 else "w"
    f_out = open(OUTPUT_FILE, mode, newline="", encoding="utf-8")
    writer = csv.writer(f_out)

    if mode == "w":
        writer.writerow(["instruction", "output"])

    tasks = []
    batch_size = 50

    print("🚀 开始处理...")

    for i, code in enumerate(data_list):
        # 跳过已处理
        if i < processed_count:
            continue

        # 跳过过长的代码
        if len(code) > 6000: continue

        # 创建任务
        task = asyncio.create_task(generate_instruction(code))
        tasks.append(task)

        # 凑够一批，或者到最后一条了，开始执行
        if len(tasks) >= batch_size or i == len(data_list) - 1:

            # === 【关键修复】使用 asyncio.as_completed + tqdm ===
            # 这种写法兼容性最好，不会报 'await' 错误
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {i // batch_size}"):
                # 获取结果（因为我们修改了函数，现在它返回两个值）
                result, original_code = await f

                if result:
                    writer.writerow([result, original_code])

            # 这一批处理完，立即刷入硬盘
            f_out.flush()
            tasks = []  # 清空任务列表

    f_out.close()
    print("🎉 所有数据处理完成！")


if __name__ == "__main__":
    # Windows 必须加这一行
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止脚本。")