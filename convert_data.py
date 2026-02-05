import csv
import json
import os

# 定义文件路径
csv_file_path = 'data/true_translated_text_cut100.csv'
json_file_path = 'data/hksl_train_cut100.json'

# 定义固定的指令
instruction_text = "将以下金融粤语文本转换为符合香港手语(HKSL)语法的Gloss格式，仅输出结果。"

data_list = []

# 读取 CSV
# encoding='utf-8-sig' 可以处理 Excel 保存 CSV 时可能带有的 BOM 头
try:
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # 获取 CSV 里的内容，如果列名有空格会自动去除
            input_text = row.get('input_text', '').strip()
            translated_text = row.get('translated_text', '').strip()
            
            # 只有当输入和输出都不为空时才添加
            if input_text and translated_text:
                data_list.append({
                    "instruction": instruction_text,
                    "input": input_text,
                    "output": translated_text
                })

    # 写入 JSON
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=2)

    print(f"转换成功！共处理了 {len(data_list)} 条数据。")
    print(f"文件已保存为: {json_file_path}")

except FileNotFoundError:
    print(f"错误：找不到文件 {csv_file_path}，请确认文件名和路径。")
except Exception as e:
    print(f"发生错误: {e}")