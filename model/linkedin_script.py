#!/usr/bin/env python3
"""
Скрипт для сбора данных о компаниях с использованием LinkedIn API
"""

import sys
import json
import os
from datetime import datetime

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Создаем клиент API
client = ApiClient()

# Список компаний для сбора данных
companies = [
    "microsoft",
    "apple",
    "google",
    "amazon",
    "meta",
    "tesla",
    "nvidia",
    "intel",
    "jpmorgan",
    "bankofamerica",
    "wellsfargo",
    "goldmansachs",
    "visa",
    "mastercard",
    "johnson",
    "pfizer",
    "unitedhealth",
    "merck",
    "procter-gamble",
    "coca-cola",
    "pepsico",
    "walmart",
    "mcdonalds",
    "generalelectric",
    "boeing",
    "caterpillar",
    "3m",
    "exxonmobil",
    "chevron",
    "ibm",
    "oracle",
    "salesforce",
    "adobe",
    "cisco",
    "deloitte",
    "mckinsey",
    "bcg",
    "accenture",
    "kpmg",
    "pwc",
    "ey"
]

# Создаем директорию для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
linkedin_dir = os.path.join(output_dir, "linkedin_data")
os.makedirs(linkedin_dir, exist_ok=True)

# Функция для сохранения данных в JSON-файл
def save_to_json(data, directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

# Сбор данных о компаниях
collected_data = []

for company_name in companies:
    print(f"Сбор данных для компании: {company_name}")
    
    try:
        # Получение данных о компании из LinkedIn
        company_data = client.call_api('LinkedIn/get_company_details', query={'username': company_name})
        
        # Проверка успешности запроса
        if company_data.get('success', False):
            filename = f"{company_name}_linkedin.json"
            filepath = save_to_json(company_data, linkedin_dir, filename)
            print(f"  Данные сохранены в: {filepath}")
            
            # Добавляем информацию в список собранных данных
            collected_data.append({
                "company_name": company_name,
                "linkedin_id": company_data.get('data', {}).get('id', ''),
                "file": filename
            })
        else:
            print(f"  Ошибка при получении данных для {company_name}: {company_data.get('message', 'Неизвестная ошибка')}")
    
    except Exception as e:
        print(f"  Ошибка при сборе данных для {company_name}: {str(e)}")

# Сохраняем метаданные о собранных данных
metadata = {
    "collection_date": datetime.now().isoformat(),
    "companies_count": len(companies),
    "successful_collections": len(collected_data),
    "companies": collected_data
}

metadata_path = save_to_json(metadata, output_dir, "linkedin_metadata.json")
print(f"\nМетаданные сохранены в: {metadata_path}")
print(f"Всего собрано данных о компаниях: {len(collected_data)}")
