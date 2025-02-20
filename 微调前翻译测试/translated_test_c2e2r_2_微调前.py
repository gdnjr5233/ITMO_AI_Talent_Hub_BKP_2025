import os
import re
import csv
from transformers import MarianMTModel, MarianTokenizer

# Загрузите модель перевода
tokenizer_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model_zh_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tokenizer_en_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model_en_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

def extract_comments_from_file(file_path):

    """Из файла Python извлекать односторонние и многополосные комментарии, записывая при этом типы комментариев"""

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Совпадают с однополосными комментариями
    single_line_pattern = r"#(.+)"
    single_line_matches = re.finditer(single_line_pattern, code)
    single_line_comments = [(match.group(1).strip(), "Single-line") for match in single_line_matches]
    
    # Совпадают с многополосными комментариями
    multi_line_pattern = r"['\"]{3}([\s\S]*?)['\"]{3}"
    multi_line_matches = re.finditer(multi_line_pattern, code)
    multi_line_comments = [(match.group(1).strip(), "Multi-line") for match in multi_line_matches]
    
    # Сочетание одноколейных и многополосных примечаний
    comments = single_line_comments + multi_line_comments
    return code, comments

def translate_line_by_line(text, target_language_model, target_language_tokenizer):

    """Перевод примечаний по строкам, сохранение прежних сокращений и изменений"""

    lines = text.split("\n")
    translated_lines = []

    for line in lines:
        leading_whitespace = len(line) - len(line.lstrip())  # Приобретение сокращенного числа пустот
        stripped_line = line.strip()  # Удалить верхнюю пустоту
        if stripped_line:  # Если это не пустая дорожка
            inputs = target_language_tokenizer(stripped_line, return_tensors="pt", truncation=True)
            outputs = target_language_model.generate(**inputs)
            translated_line = target_language_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Добавлены оригинальные аббревиатуры
            translated_lines.append(" " * leading_whitespace + translated_line)
        else:
            translated_lines.append("")  # сохраняя пустоту

    return "\n".join(translated_lines)

# def translate_line_by_line(text, target_language_model, target_language_tokenizer):
    
#     """逐行翻译注释，保持换行和缩进"""
    
#     lines = text.split("\n")  # Расчленение многомерных примечаний
#     translated_lines = []

#     for line in lines:
#         stripped_line = line.strip()  # Удалить верхнюю пустоту
#         if stripped_line:  # Если это не пустая дорожка
#             inputs = target_language_tokenizer(stripped_line, return_tensors="pt", truncation=True)
#             outputs = target_language_model.generate(**inputs)
#             translated_line = target_language_tokenizer.decode(outputs[0], skip_special_tokens=True)
#             translated_lines.append(translated_line)
#         else:
#             translated_lines.append("")  # сохраняя пустоту
    
#     # Удерживайте скидку и замену исходной строки
#     return "\n".join(translated_lines)

def translate_text(text):
    
    """Китайский -> Английский -> Русский язык, перевод по строкам, формат мульти-линейных комментариев"""
    
    # Китайский -> Перевод на английский язык
    translated_en = translate_line_by_line(text, model_zh_en, tokenizer_zh_en)
    
    # Английский -> Русский перевод по очереди
    translated_ru = translate_line_by_line(translated_en, model_en_ru, tokenizer_en_ru)
    
    return translated_en, translated_ru

def process_single_file(file_path, output_csv_path):

    """Обрабатывать отдельные файлы Python, извлекать примечания, переводить и сохранять в файлах CSV"""

    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "code_comment_type", "comment_zh", "comment_ru", "comment_en"])  # CSV 表头
        
        # Извлечение кода и примечаний
        code, comments = extract_comments_from_file(file_path)
        
        for comment, comment_type in comments:
            translated_en, translated_ru = translate_text(comment)  # Переводные примечания
            writer.writerow([file_path, code, comment_type, comment, translated_ru, translated_en])  # Запись в КСВ

def process_files_in_folder(folder_path, output_csv_path):
    
    """Отталкиваясь от всех файлов Python в папках, извлекая примечания, переводя и сохраняя их в файлы CSV"""
    
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "code", "code_comment_type", "comment_zh", "comment_ru", "comment_en"])  # CSV 表头
        
        # Категория: Персональные папки
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):  # Обрабатывается только .py-файлы
                    file_path = os.path.join(root, file)

                    # Извлечение кода и примечаний
                    code, comments = extract_comments_from_file(file_path)
                    
                    for comment, comment_type in comments:
                        translated_en, translated_ru = translate_text(comment)  # Переводные примечания
                        writer.writerow([file_path, code, comment_type, comment, translated_ru, translated_en])  # Запись в КСВ

# Путь папки для примера
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes"  # Заменить путь для вашей папки
# folder_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes2"

# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\dataset_v1.csv"  # Замена на ваш выходной путь
# output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\dataset_v2.csv"

# Примерная траектория файла
file_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\pythoncodes\推荐系统_归一化折扣累计增益.py"
output_csv_path = r"C:\Users\gdnjr5233_YOLO\Desktop\ВКР_2025\datasets\推荐系统_归一化折扣累计增益.csv"

# Обрабатывая отдельные файлы и генерируя CSV
process_single_file(file_path, output_csv_path)
print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")

# Обрабатывая папки и генерируя CSV
# process_files_in_folder(folder_path, output_csv_path)
# print(f"注释翻译已保存到 CSV 文件：{output_csv_path}")
