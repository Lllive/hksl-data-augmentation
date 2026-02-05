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

# 加载环境变量
load_dotenv()

# ================= 1. 配置日志 =================
logging.basicConfig(
    filename='api_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= 2. 配置区域 =================

TEST_MODE = False      # 保持测试模式
TEST_LIMIT = 5        # 设为 5 条即可，方便看日志
INPUT_FILE = 'data/true_translated_text_cut100.csv' 
OUTPUT_FILE = 'augmented_dataset.csv'

# 配置 Qwen 模型信息
MODELS_CONFIG = [
    {   
        "name": "qwen3-instruct",
        "url": os.getenv("OPENAI_API_URL_QWEN"), 
        "key": os.getenv("OPENAI_API_KEY_QWEN"), 
        "model_id": "qwen-plus", 
        "params": {
            "temperature": 0.7,
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
def call_llm_with_retry(original_zh, original_hksl):
    """
    调用 LLM 生成改写数据 (强制繁体中文版)
    """
    
    # --- 1. System Prompt ---
    system_content = (
        "你是一个专业的中文数据增强专家。\n"
        "⚠️⚠️⚠️ 核心规则：所有输出内容（尤其是 'zh' 字段）必须严格使用繁体中文 (Traditional Chinese)，绝对禁止使用简体字。⚠️⚠️⚠️\n"
        "同时，你只输出纯 JSON 数组，不要包含任何其他废话。"
    )

    # --- 2. User Prompt ---
    prompt = f"""
    任务：将以下“中文输入”改写成 3 种不同说法（如：口语化、书面化、倒装），保持“HKSL输出”不变。
    
    【原始数据】：
    中文输入：{original_zh}
    HKSL输出：{original_hksl}

    【严格输出格式】：
    1. 必须是纯 JSON 列表。
    2. 列表中的每个对象必须严格包含两个键："zh" 和 "hksl"。
    3. "zh" 对应的值必须是【繁体中文】。
    4. 不要输出 markdown 标记（如 ```json），直接输出内容。
    
    示例格式（请参考此繁体格式）：
    [
        {{"zh": "改寫後的繁體中文句子1", "hksl": "{original_hksl}"}},
        {{"zh": "改寫後的繁體中文句子2", "hksl": "{original_hksl}"}},
        {{"zh": "改寫後的繁體中文句子3", "hksl": "{original_hksl}"}}
    ]
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
        
        # ==========================================
        # 🖨️ [新增功能] 打印模型原始输出
        # ==========================================
        print(f"\n{'='*20} 模型原始输出 {'='*20}")
        print(content)
        print(f"{'='*50}\n")
        # ==========================================
        # --- 数据清洗 ---
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        
        # --- 3. JSON 解析 (这里是关键修复点) ---
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败，原始内容: {content}")
            return [] # 解析失败返回空列表，触发重试或跳过

        # --- 4. 标准化为列表 ---
        final_list = []
        if isinstance(data, dict):
            # 尝试寻找列表字段 (防止模型包了一层 {"data": [...]})
            for key in data:
                if isinstance(data[key], list):
                    final_list = data[key]
                    break
            # 如果没找到列表，本身可能就是单条数据对象
            if not final_list: 
                final_list = [data]
        elif isinstance(data, list):
            final_list = data
            
        return final_list # ✅ 必须返回 List

    except Exception as e:
        print(f"❌ API调用异常: {e}")
        logging.warning(f"API调用异常: {e}")
        raise e # 抛出异常以触发 @retry

# ================= 4. 主程序逻辑 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        current_output_file = OUTPUT_FILE
        if TEST_MODE:
            print(f"\n⚠️  注意：当前为【测试模式】，仅处理前 {TEST_LIMIT} 条")
            df = df.head(TEST_LIMIT)
            current_output_file = "test_" + OUTPUT_FILE
        
        print(f"✅ 开始处理...\n")

    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    all_data = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="AI处理中"):
        original_zh = row['input_text']
        original_hksl = row['translated_text']
        
        # 保存原始数据
        all_data.append({
            "chinese": original_zh, 
            "hksl": original_hksl, 
            "type": "original"
        })
        
        try:
            augmented_list = call_llm_with_retry(original_zh, original_hksl)
            
            if augmented_list:
                success_count = 0
                for item in augmented_list:
                    # --- 🛠️ 修复点：更灵活的 Key 查找 ---
                    # 尝试找中文 Key：zh, Chinese, chinese, input
                    zh_val = item.get('zh') or item.get('chinese') or item.get('Chinese') or item.get('input')
                    # 尝试找手语 Key：hksl, HKSL, output
                    hksl_val = item.get('hksl') or item.get('HKSL') or item.get('output')

                    if zh_val and hksl_val:
                        all_data.append({
                            "chinese": zh_val, 
                            "hksl": hksl_val,
                            "type": "augmented"
                        })
                        success_count += 1
                    else:
                        print(f"⚠️ 数据格式不符，丢弃: {item}")
                
                # print(f"  -> 成功生成 {success_count} 条增强数据") # 调试用
            else:
                print(f"⚠️ 返回为空列表")

        except Exception as e:
            logging.error(f"处理失败: {e}")
            continue
            
        # 定期保存
        if (index + 1) % 5 == 0:
            pd.DataFrame(all_data).to_csv("test_temp_backup.csv", index=False, encoding='utf-8-sig')

    # 最终保存
    final_df = pd.DataFrame(all_data)
    final_df.to_csv(current_output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 处理完成！")
    print(f"原始数据: {len(df)} 条")
    print(f"最终数据: {len(final_df)} 条 (如果这个数字等于原始数据，说明增强全失败了)")
    print(f"结果保存至: {current_output_file}")

if __name__ == "__main__":
    main()