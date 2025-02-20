import os
import re
import csv
from transformers import MarianMTModel, MarianTokenizer

# Загрузите две модели перевода
tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model_zh_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tokenizer_en_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model_en_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

def extract_comments_from_file(file_path):
    """Из файла Python извлекать односторонние и многополосные примечания"""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Совпадают с однополосными комментариями
    single_line_pattern = r"#(.+)"
    single_line_comments = re.findall(single_line_pattern, code)
    
    # Совпадают с многополосными комментариями
    multi_line_pattern = r"['\"]{3}([\s\S]*?)['\"]{3}"
    multi_line_comments = re.findall(multi_line_pattern, code)
    
    # Сочетание одноколейных и многополосных примечаний
    comments = single_line_comments + multi_line_comments
    return code, [comment.strip() for comment in comments if comment.strip()]

def translate_text(text):
    """Китайский -> Английский -> Русский"""
    # Китайско -> Английский
    inputs = tokenizer_zh_en(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_zh_en.generate(**inputs)
    translated_en = tokenizer_zh_en.decode(outputs[0], skip_special_tokens=True)
    
    # Английский -> Русский
    inputs = tokenizer_en_ru(translated_en, return_tensors="pt", padding=True, truncation=True)
    outputs = model_en_ru.generate(**inputs)
    translated_ru = tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
    
    return translated_en, translated_ru

def process_files_in_folder(folder_path, output_csv_path):
    """Отталкиваясь от всех файлов Python в папках, извлекая примечания, переводя и сохраняя их в файлы CSV"""
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "comment_zh", "comment_en", "comment_ru"])  # CSV 表头
        
        # Категория: Персональные папки
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):  # Обрабатывается только .py-файлы
                    file_path = os.path.join(root, file)
                    # print(f"Processing file: {file_path}")
                    
                    # Извлечение кода и примечаний
                    code, comments = extract_comments_from_file(file_path)
                    
                    for comment in comments:
                        translated_en, translated_ru = translate_text(comment)  # Переводные примечания
                        writer.writerow([file_path, code, comment, translated_en, translated_ru])  # Запись в КСВ
                        # print(f"Processed comment: {comment} -> {translated_en} -> {translated_ru}")

# Путь папки для примера
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythonCodes"  # Заменить путь для вашей папки
folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\medium"

# Выходные пути файлов CSV
# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\datasets.csv"  # Замена на ваш выходной путь
output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\medium.csv"

# Обрабатывая папки и генерируя CSV
process_files_in_folder(folder_path, output_csv_path)

print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")
