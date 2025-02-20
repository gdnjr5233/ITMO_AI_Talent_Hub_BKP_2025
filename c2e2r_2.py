import os
import re
import csv
from transformers import MarianMTModel, MarianTokenizer

# 加载翻译模型
tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model_zh_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tokenizer_en_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model_en_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

def extract_comments_from_file(file_path):

    """从 Python 文件中提取单行和多行注释，同时记录注释类型"""

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 匹配单行注释
    single_line_pattern = r"#(.+)"
    single_line_matches = re.finditer(single_line_pattern, code)
    single_line_comments = [(match.group(1).strip(), "Single-line") for match in single_line_matches]
    
    # 匹配多行注释
    multi_line_pattern = r"['\"]{3}([\s\S]*?)['\"]{3}"
    multi_line_matches = re.finditer(multi_line_pattern, code)
    multi_line_comments = [(match.group(1).strip(), "Multi-line") for match in multi_line_matches]
    
    # 合并单行和多行注释
    comments = single_line_comments + multi_line_comments
    return code, comments

def translate_line_by_line(text, target_language_model, target_language_tokenizer):

    """逐行翻译注释，保持原有缩进和换行"""

    lines = text.split("\n")
    translated_lines = []

    for line in lines:
        leading_whitespace = len(line) - len(line.lstrip())  # 获取缩进的空格数
        stripped_line = line.strip()  # 去掉首尾空格
        if stripped_line:  # 如果这一行不是空行
            inputs = target_language_tokenizer(stripped_line, return_tensors="pt", truncation=True)
            outputs = target_language_model.generate(**inputs)
            translated_line = target_language_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 添加原来的缩进
            translated_lines.append(" " * leading_whitespace + translated_line)
        else:
            translated_lines.append("")  # 保留空行

    return "\n".join(translated_lines)

# def translate_line_by_line(text, target_language_model, target_language_tokenizer):
    
#     """逐行翻译注释，保持换行和缩进"""
    
#     lines = text.split("\n")  # 分割多行注释
#     translated_lines = []

#     for line in lines:
#         stripped_line = line.strip()  # 去掉首尾空格
#         if stripped_line:  # 如果这一行不是空行
#             inputs = target_language_tokenizer(stripped_line, return_tensors="pt", truncation=True)
#             outputs = target_language_model.generate(**inputs)
#             translated_line = target_language_tokenizer.decode(outputs[0], skip_special_tokens=True)
#             translated_lines.append(translated_line)
#         else:
#             translated_lines.append("")  # 保持空行
    
#     # 保留缩进和原始行的换行
#     return "\n".join(translated_lines)

def translate_text(text):
    
    """中文 -> 英文 -> 俄文，逐行翻译，保持多行注释格式"""
    
    # 中文 -> 英文逐行翻译
    translated_en = translate_line_by_line(text, model_zh_en, tokenizer_zh_en)
    
    # 英文 -> 俄文逐行翻译
    translated_ru = translate_line_by_line(translated_en, model_en_ru, tokenizer_en_ru)
    
    return translated_en, translated_ru

def process_single_file(file_path, output_csv_path):

    """处理单个 Python 文件，提取注释、翻译并保存到 CSV 文件"""

    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "code_comment_type", "comment_zh", "comment_ru", "comment_en"])  # CSV 表头
        
        # 提取代码和注释
        code, comments = extract_comments_from_file(file_path)
        
        for comment, comment_type in comments:
            translated_en, translated_ru = translate_text(comment)  # 翻译注释
            writer.writerow([file_path, code, comment_type, comment, translated_ru, translated_en])  # 写入 CSV

def process_files_in_folder(folder_path, output_csv_path):
    
    """递归处理文件夹中的所有 Python 文件，提取注释、翻译并保存到 CSV 文件"""
    
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "code_comment_type", "comment_zh", "comment_ru", "comment_en"])  # CSV 表头
        
        # 遍历文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):  # 只处理 .py 文件
                    file_path = os.path.join(root, file)

                    # 提取代码和注释
                    code, comments = extract_comments_from_file(file_path)
                    
                    for comment, comment_type in comments:
                        translated_en, translated_ru = translate_text(comment)  # 翻译注释
                        writer.writerow([file_path, code, comment_type, comment, translated_ru, translated_en])  # 写入 CSV

# 示例文件夹路径
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes"  # 替换为你的文件夹路径
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes2"

# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\dataset_v1.csv"  # 替换为你的输出路径
# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\dataset_v2.csv"

# 示例文件路径
file_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes\推荐系统_归一化折扣累计增益.py"
output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\推荐系统_归一化折扣累计增益.csv"

# 处理单个文件并生成 CSV
process_single_file(file_path, output_csv_path)
print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")

# 处理文件夹并生成 CSV
# process_files_in_folder(folder_path, output_csv_path)
# print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")
