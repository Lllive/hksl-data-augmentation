import pandas as pd
import opencc
import os

def process_excel(input_file, output_file):
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    print(f"正在读取文件: {input_file} ...")
    
    # 2. 读取 Excel 中的所有 Sheet (sheet_name=None 会返回一个字典)
    try:
        all_sheets = pd.read_excel(input_file, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 3. 初始化繁转简转换器 (t2s: Traditional to Simplified)
    cc = opencc.OpenCC('t2s')
    
    # 准备一个列表来收集处理后的数据
    processed_sheets = {}

    # 目标列名 (根据你的截图，列名是 "Words")
    target_col = "Words"

    # 4. 遍历每一个 Sheet 进行处理
    for sheet_name, df in all_sheets.items():
        print(f"正在处理 Sheet: {sheet_name} ...")
        
        # 检查目标列是否存在
        if target_col in df.columns:
            # --- 步骤 A: 繁体转简体 ---
            # 使用 apply 函数应用转换，同时处理非字符串的情况（防止报错）
            def convert_text(text):
                if pd.isna(text): # 如果是空值，保持原样
                    return text
                return cc.convert(str(text))
            
            df[target_col] = df[target_col].apply(convert_text)
            
            # --- 步骤 B: 剔除长度为 1 的词 ---
            # 计算长度
            lengths = df[target_col].astype(str).str.len()
            
            # 记录删除前的行数
            original_count = len(df)
            
            # 只保留长度不等于 1 的行 (即剔除 length == 1)
            # 注意：这里也会保留空值或长度为0的值，如果你只想保留长度>1，可以改为 > 1
            df = df[lengths != 1]
            
            removed_count = original_count - len(df)
            print(f"  - 已转换繁体为简体")
            print(f"  - 已剔除 {removed_count} 行长度为1的数据")
            
        else:
            print(f"  - 警告: 在该 Sheet 中未找到列名 '{target_col}'，跳过处理。")
        
        # 将处理后的 df 存入字典
        processed_sheets[sheet_name] = df

    # 5. 保存到新的 Excel 文件
    print(f"正在保存到: {output_file} ...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print("处理完成！")

# --- 执行配置 ---
if __name__ == "__main__":
    # 输入文件名 (请确保文件名和路径正确)
    input_filename = 'filtered.xlsx'
    
    # 输出文件名 (建议另存为新文件，防止覆盖原文件出错)
    output_filename = 'filtered_processed.xlsx'
    
    process_excel(input_filename, output_filename)