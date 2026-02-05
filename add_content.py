import pandas as pd
import time
import os
import json
import logging
import re
from dotenv import load_dotenv
from openai import OpenAI, APIError
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= 0. 加载环境变量 =================
load_dotenv()

# ================= 1. 配置日志 =================
logging.basicConfig(
    filename='context_generation_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= 2. 配置区域 =================
TEST_MODE = False        # ⚠️ 调试完成后，记得改为 False
TEST_LIMIT = 5           
SAMPLE_SIZE = 1200       # 随机抽取 1200 条

# 输入文件
INPUT_FILE = 'augmented_dataset.csv' 
OUTPUT_FILE = 'dataset_with_context_v2.csv'

# 配置 Qwen 模型信息
MODELS_CONFIG = [
    {
        "name": "qwen3-instruct",
        "url": os.getenv("OPENAI_API_URL_QWEN"),
        "key": os.getenv("OPENAI_API_KEY_QWEN"),
        "model_id": "qwen-plus",
        "params": {
            "temperature": 0.8,
            "max_tokens": 1000,
            "top_p": 0.8
        }
    },
]

active_config = MODELS_CONFIG[0]
client = OpenAI(
    api_key=active_config['key'],
    base_url=active_config['url']
)

# ================= 3. API 调用函数 =================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_context_with_retry(original_text):
    """
    调用 LLM 为句子生成场景上下文
    """
    system_content = (
        "你是一个繁体中文数据增强助手。\n"
        "你的任务是为句子添加简短的发生场景（Context）。\n"
        "请严格输出 JSON 格式。"
    )

    prompt = f"""
    任务：请为这句话构想一个简短的发生场景（Context），并将其加在句子前面，用括号标注。
    
    【输入句子】：{original_text}

    【要求】：
    1. 场景描述要简短，不超过 10 个字。
    2. 必须保持繁体中文。
    3. 严格输出 JSON 对象，包含 key "new_sentence"。

    【示例】：
    输入：多少钱？
    输出：{{"new_sentence": "(在菜市場買菜時) 多少錢？"}}

    输入：医生我头痛
    输出：{{"new_sentence": "(在醫院) 醫生我頭痛"}}
    
    请输出 JSON：
    """

    try:
        response = client.chat.completions.create(
            model=active_config['model_id'],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            **active_config['params']
        )
        
        content = response.choices[0].message.content
        # 清理可能存在的 markdown 标记
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            return data.get("new_sentence", None)
        except json.JSONDecodeError:
            # 如果解析失败，打印原始内容以便调试
            tqdm.write(f"⚠️ JSON 解析失败，模型返回内容: {content}")
            return None

    except Exception as e:
        tqdm.write(f"❌ API调用异常: {e}")
        raise e 

# ================= 4. 主程序逻辑 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        
        if 'chinese' not in df.columns:
            print(f"❌ 错误：CSV中找不到 'chinese' 列。当前列名: {df.columns}")
            return

        print(f"✅ 原始数据加载成功，共 {len(df)} 条")

        if TEST_MODE:
            print(f"\n⚠️  测试模式：仅处理前 {TEST_LIMIT} 条")
            subset = df.head(TEST_LIMIT).copy()
        else:
            if SAMPLE_SIZE and SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(df):
                print(f"🎲 正在随机抽取 {SAMPLE_SIZE} 条数据进行增强...")
                subset = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
            else:
                print("Processing all data...")
                subset = df.copy()

    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    new_rows = []
    print(f"🚀 开始处理 {len(subset)} 条数据...\n")

    # 使用 tqdm 显示进度条
    for index, row in tqdm(subset.iterrows(), total=len(subset), desc="AI生成场景中"):
        original_zh = row['chinese']
        original_hksl = row['hksl'] if 'hksl' in row else "" 
        
        try:
            new_zh_with_context = generate_context_with_retry(original_zh)
            
            if new_zh_with_context:
                # ==========================================
                # 👇 修改点：在这里打印生成结果，让你能看见！
                # ==========================================
                tqdm.write(f"✨ [原句] {original_zh}")
                tqdm.write(f"✅ [新句] {new_zh_with_context}")
                tqdm.write("-" * 40) # 分割线
                
                new_row = {
                    'chinese': new_zh_with_context,
                    'hksl': original_hksl,
                    'type': 'context_expanded'
                }
                new_rows.append(new_row)
            else:
                logging.error(f"生成返回空值: {original_zh}")

        except Exception as e:
            logging.error(f"处理行 {index} 失败: {e}")
            continue
            
        # 定期保存
        if len(new_rows) > 0 and len(new_rows) % 50 == 0:
            pd.DataFrame(new_rows).to_csv("temp_context_backup.csv", index=False, encoding='utf-8-sig')

    # ================= 5. 合并与保存 =================
    
    print("\n💾 正在合并数据并保存...")
    
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_final = pd.concat([df, df_new], ignore_index=True)
        
        df_final.to_csv(OUTPUT_FILE, index=False, header=True, encoding='utf-8-sig')
        
        print("=======================================")
        print(f"🎉 处理完成！")
        print(f"原数据: {len(df)} 条")
        print(f"新增数据: {len(df_new)} 条")
        print(f"最终数据: {len(df_final)} 条")
        print(f"结果已保存至: {OUTPUT_FILE}")
    else:
        print("⚠️ 未生成任何新数据。")

if __name__ == "__main__":
    main()