import os
import re
import csv
from transformers import MarianMTModel, MarianTokenizer

# 加载两个翻译模型
tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model_zh_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tokenizer_en_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model_en_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

def extract_comments_from_file(file_path):
    """从 Python 文件中提取单行和多行注释"""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 匹配单行注释
    single_line_pattern = r"#(.+)"
    single_line_comments = re.findall(single_line_pattern, code)
    
    # 匹配多行注释
    multi_line_pattern = r"['\"]{3}([\s\S]*?)['\"]{3}"
    multi_line_comments = re.findall(multi_line_pattern, code)
    
    # 合并单行和多行注释
    comments = single_line_comments + multi_line_comments
    return code, [comment.strip() for comment in comments if comment.strip()]

def translate_text(text):
    """中文 -> 英文 -> 俄文"""
    # 中文 -> 英文
    inputs = tokenizer_zh_en(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_zh_en.generate(**inputs)
    translated_en = tokenizer_zh_en.decode(outputs[0], skip_special_tokens=True)
    
    # 英文 -> 俄文
    inputs = tokenizer_en_ru(translated_en, return_tensors="pt", padding=True, truncation=True)
    outputs = model_en_ru.generate(**inputs)
    translated_ru = tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
    
    return translated_en, translated_ru

def process_files_in_folder(folder_path, output_csv_path):
    """递归处理文件夹中的所有 Python 文件，提取注释、翻译并保存到 CSV 文件"""
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "comment_zh", "comment_en", "comment_ru"])  # CSV 表头
        
        # 遍历文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):  # 只处理 .py 文件
                    file_path = os.path.join(root, file)
                    # print(f"Processing file: {file_path}")
                    
                    # 提取代码和注释
                    code, comments = extract_comments_from_file(file_path)
                    
                    for comment in comments:
                        translated_en, translated_ru = translate_text(comment)  # 翻译注释
                        writer.writerow([file_path, code, comment, translated_en, translated_ru])  # 写入 CSV
                        # print(f"Processed comment: {comment} -> {translated_en} -> {translated_ru}")

# 示例文件夹路径
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythonCodes"  # 替换为你的文件夹路径
folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\medium"

# 输出 CSV 文件路径
# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\datasets.csv"  # 替换为你的输出路径
output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\medium.csv"

# 处理文件夹并生成 CSV
process_files_in_folder(folder_path, output_csv_path)

print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")
