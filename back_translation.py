import pandas as pd
import os
import logging
import json
import re
import difflib
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 0. 加载环境变量 =================
load_dotenv()

# ================= 1. 配置日志 =================
logging.basicConfig(
    filename='multilang_backtrans_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= 2. 配置区域 =================
# --- 🛠️ 测试模式开关 ---
TEST_MODE = True       # True: 仅跑几条测试; False: 跑全量
TEST_LIMIT = 5         # 测试模式下处理的 original 数据条数

INPUT_FILE = 'dataset_with_context_v2.csv'  # 输入文件
OUTPUT_FILE = 'dataset_backtranslated.csv'  # 输出文件
REJECTED_FILE = 'backtrans_rejected.csv'

# --- 阈值设置 ---
MAX_TEXT_SIMILARITY = 0.95     
MIN_SEMANTIC_SIMILARITY = 0.85 
MIN_LEN_RATIO = 0.6    
MAX_LEN_RATIO = 2.0    

MAX_WORKERS = 8  

if TEST_MODE:
    print(f"\n⚠️  注意：当前为【测试模式】，仅处理前 {TEST_LIMIT} 条 'original' 数据。")
    print(f"⚠️  并发数将强制调整为 1，以便在控制台查看打印输出。\n")
    MAX_WORKERS = 1
    OUTPUT_FILE = 'test_output_backtranslated.csv' # 测试结果存到不同文件

MODELS_CONFIG = [
    {
        "name": "qwen3-instruct",
        "url": os.getenv("OPENAI_API_URL_QWEN"), 
        "key": os.getenv("OPENAI_API_KEY_QWEN"),
        "model_id": "qwen-plus", 
        "params": {
            "temperature": 0.7,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }
    },
]
active_config = MODELS_CONFIG[0]
client = OpenAI(api_key=active_config['key'], base_url=active_config['url'])

# ================= 3. 模型加载 =================
print("⏳ 正在加载语义匹配模型...")
semantic_model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("✅ 模型加载完成！")

# ================= 4. 核心功能函数 =================

def calculate_text_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def calculate_semantic_similarity(s1, s2):
    embeddings1 = semantic_model.encode(s1, convert_to_tensor=True)
    embeddings2 = semantic_model.encode(s2, convert_to_tensor=True)
    return util.cos_sim(embeddings1, embeddings2).item()

def check_quality(original, new_text):
    if not new_text or len(new_text.strip()) == 0:
        return False, "空结果"
    
    # 1. 长度检查
    len_ratio = len(new_text) / len(original)
    if len_ratio < MIN_LEN_RATIO: return False, f"太短 ({len_ratio:.2f})"
    if len_ratio > MAX_LEN_RATIO: return False, f"太长 ({len_ratio:.2f})"

    # 2. 字面相似度
    text_sim = calculate_text_similarity(original, new_text)
    if text_sim > MAX_TEXT_SIMILARITY:
        return False, f"字面太像原句 ({text_sim:.2f})"

    # 3. 语义相似度
    sem_sim = calculate_semantic_similarity(original, new_text)
    if sem_sim < MIN_SEMANTIC_SIMILARITY:
        return False, f"意思偏差 ({sem_sim:.2f})"

    return True, f"通过 (语义:{sem_sim:.2f})"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_backtrans_json(original_text):
    """
    让模型模拟多语言回译过程，直接返回最终中文结果
    """
    system_content = "你是一个精通多国语言的翻译专家。请严格输出 JSON 格式。"
    
    prompt = f"""
    请对以下中文句子进行 3 种不同路径的“回译”（Back-Translation），以获得多样化的中文表达。
    
    原始中文："{original_text}"

    请执行以下步骤（在内部思考，只输出最终的中文结果）：
    1. 路径A：中文 -> 英文 -> 中文
    2. 路径B：中文 -> 德文 -> 中文 (利用德语语序差异重组句子)
    3. 路径C：中文 -> 日文 -> 中文 (利用日语语境差异重构句子)

    要求：
    - 最终输出的中文必须通顺、自然。
    - 意思必须与原句完全一致（因为要对应手语）。
    - 尽量与原句的字面措辞有所不同。

    【请输出 JSON 格式】：
    {{
        "variants": [
            {{"zh": "路径A的结果", "path": "zh-en-zh"}},
            {{"zh": "路径B的结果", "path": "zh-de-zh"}},
            {{"zh": "路径C的结果", "path": "zh-ja-zh"}}
        ]
    }}
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
        content = response.choices[0].message.content.strip()
        
        # --- 测试模式：打印原始输出 ---
        if TEST_MODE:
            print(f"\n[TEST] 原句: {original_text}")
            print(f"[TEST] 模型返回: {content}\n")
        # ---------------------------

        try:
            clean_content = re.sub(r'```json\s*|\s*```', '', content)
            data = json.loads(clean_content)
            return data.get("variants", [])
        except json.JSONDecodeError:
            logging.error(f"JSON解析失败: {content}")
            return []
            
    except Exception as e:
        raise e 

def process_single_row(index, row):
    original_zh = row['chinese']
    original_hksl = row['hksl']
    
    generated_results = []
    rejected_logs = []
    
    try:
        # 调用多语言回译
        variants_data = generate_backtrans_json(original_zh)
        
        for item in variants_data:
            new_zh = item.get('zh', '').strip()
            path_type = item.get('path', 'unknown')
            
            if not new_zh: continue

            # 质量检测
            is_valid, reason = check_quality(original_zh, new_zh)
            
            if is_valid:
                generated_results.append({
                    'chinese': new_zh,
                    'hksl': original_hksl, 
                    'type': f'backtrans_{path_type}' 
                })
            else:
                rejected_logs.append({
                    'original': original_zh,
                    'generated': new_zh,
                    'path': path_type,
                    'reason': reason
                })
                
    except Exception as e:
        logging.error(f"行 {index} 处理失败: {e}")
            
    return generated_results, rejected_logs

# ================= 5. 主程序 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    # 1. 读取原始文件
    df = pd.read_csv(INPUT_FILE)
    print(f"📄 读取文件成功，总行数: {len(df)}")
    
    # 2. 【关键修改】只筛选 type 为 'original' 的行
    # 这样可以保证只对原句做增强，不会对已经是 augm 的数据做二次处理
    if 'type' not in df.columns:
        print("❌ 错误：CSV文件中缺少 'type' 列，无法筛选 original 数据。")
        return

    originals_df = df[df['type'] == 'original'].copy()
    print(f"🔍 筛选出 'original' 数据: {len(originals_df)} 条")

    # 3. 根据是否测试模式，确定最终要处理的数据 target_df
    if TEST_MODE:
        target_df = originals_df.head(TEST_LIMIT).copy()
        print(f"🧪 测试模式：仅处理前 {len(target_df)} 条数据")
    else:
        target_df = originals_df.copy()
        print(f"🚀 正式模式：将处理所有 {len(target_df)} 条 original 数据")
    
    if len(target_df) == 0:
        print("⚠️ 没有需要处理的数据，程序结束。")
        return

    all_new_rows = []
    all_rejected = []
    
    # 4. 开始多线程处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_single_row, idx, row): idx 
            for idx, row in target_df.iterrows()
        }
        
        if TEST_MODE:
            print("👀 测试模式下直接输出日志...")
            for future in as_completed(future_to_index):
                try:
                    new_rows, rejected = future.result()
                    all_new_rows.extend(new_rows)
                    all_rejected.extend(rejected)
                except Exception as exc:
                    logging.error(f"任务异常: {exc}")
        else:
            for future in tqdm(as_completed(future_to_index), total=len(target_df), desc="回译中"):
                try:
                    new_rows, rejected = future.result()
                    all_new_rows.extend(new_rows)
                    all_rejected.extend(rejected)
                except Exception as exc:
                    logging.error(f"任务异常: {exc}")

    # 5. 保存结果
    print(f"\n📊 统计:")
    print(f"  - 输入总数据: {len(df)} 条")
    print(f"  - 本次处理源数据: {len(target_df)} 条")
    print(f"  - 新增回译数据: {len(all_new_rows)} 条")
    print(f"  - 过滤掉的数据: {len(all_rejected)} 条")

    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)
        
        # 【关键合并】
        # 将 "原始的完整数据(df)" 和 "新生成的数据(df_new)" 拼在一起
        # 这样既保留了 original 原句，也保留了旧的 augm 数据，又增加了新生成的句子
        df_final = pd.concat([df, df_new], ignore_index=True)
            
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"🎉 结果已保存至新文件: {OUTPUT_FILE}")
    else:
        print("⚠️ 本次没有生成任何有效的新数据。")

    if all_rejected:
        pd.DataFrame(all_rejected).to_csv(REJECTED_FILE, index=False, encoding='utf-8-sig')
        print(f"📝 拒绝记录已保存: {REJECTED_FILE}")

if __name__ == "__main__":
    main()