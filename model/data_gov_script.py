#!/usr/bin/env python3
"""
Скрипт для поиска и сбора данных из Data.gov
"""

import requests
import json
import os
import time
from datetime import datetime

# Создаем директорию для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
data_gov_dir = os.path.join(output_dir, "data_gov_datasets")
os.makedirs(data_gov_dir, exist_ok=True)

# Базовый URL для API Data.gov
BASE_URL = "https://catalog.data.gov/api/3/action/package_search"

# Ключевые слова для поиска по областям консалтинга
search_terms = {
    "it_consulting": [
        "information technology", 
        "digital transformation", 
        "it infrastructure", 
        "cybersecurity", 
        "software development"
    ],
    "hr_consulting": [
        "human resources", 
        "workforce", 
        "employee", 
        "talent management", 
        "compensation", 
        "benefits"
    ],
    "investment_consulting": [
        "investment", 
        "venture capital", 
        "mergers acquisitions", 
        "market entry", 
        "portfolio management"
    ],
    "financial_consulting": [
        "financial planning", 
        "budgeting", 
        "cash flow", 
        "taxation", 
        "financial reporting", 
        "business valuation"
    ],
    "operations_consulting": [
        "business process", 
        "supply chain", 
        "quality control", 
        "lean manufacturing", 
        "process automation", 
        "change management"
    ],
    "management_consulting": [
        "strategic planning", 
        "organizational structure", 
        "performance management", 
        "corporate governance", 
        "risk management", 
        "leadership development"
    ]
}

# Функция для сохранения данных в JSON-файл
def save_to_json(data, directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

# Функция для поиска наборов данных
def search_datasets(query, rows=10):
    params = {
        "q": query,
        "rows": rows,
        "sort": "score desc, metadata_modified desc"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при поиске данных для запроса '{query}': {str(e)}")
        return None

# Функция для загрузки набора данных
def download_dataset(dataset_id):
    url = f"https://catalog.data.gov/api/3/action/package_show?id={dataset_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при загрузке набора данных с ID '{dataset_id}': {str(e)}")
        return None

# Сбор данных по каждой области консалтинга
collected_data = {}

for category, terms in search_terms.items():
    print(f"\nПоиск данных для категории: {category}")
    category_dir = os.path.join(data_gov_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    category_datasets = []
    
    for term in terms:
        print(f"  Поиск по запросу: '{term}'")
        search_results = search_datasets(term)
        
        if search_results and search_results.get("success", False):
            results = search_results.get("result", {}).get("results", [])
            print(f"  Найдено {len(results)} наборов данных")
            
            for dataset in results[:5]:  # Берем только первые 5 результатов для каждого запроса
                dataset_id = dataset.get("id")
                dataset_name = dataset.get("name")
                
                if dataset_id and dataset_name:
                    print(f"    Загрузка набора данных: {dataset_name}")
                    
                    # Получаем полную информацию о наборе данных
                    dataset_info = download_dataset(dataset_id)
                    
                    if dataset_info and dataset_info.get("success", False):
                        dataset_data = dataset_info.get("result")
                        
                        # Сохраняем информацию о наборе данных
                        filename = f"{dataset_name}.json"
                        filepath = save_to_json(dataset_data, category_dir, filename)
                        print(f"    Сохранено в: {filepath}")
                        
                        # Добавляем информацию в список собранных данных
                        category_datasets.append({
                            "id": dataset_id,
                            "name": dataset_name,
                            "title": dataset_data.get("title"),
                            "file": filename
                        })
                    
                    # Делаем паузу, чтобы не перегружать API
                    time.sleep(1)
        
        # Делаем паузу между запросами
        time.sleep(2)
    
    # Сохраняем информацию о собранных данных для категории
    collected_data[category] = category_datasets

# Сохраняем метаданные о собранных данных
metadata = {
    "collection_date": datetime.now().isoformat(),
    "categories": list(search_terms.keys()),
    "datasets_count": {category: len(datasets) for category, datasets in collected_data.items()},
    "total_datasets": sum(len(datasets) for datasets in collected_data.values()),
    "data": collected_data
}

metadata_path = save_to_json(metadata, data_gov_dir, "data_gov_metadata.json")
print(f"\nМетаданные сохранены в: {metadata_path}")
print(f"Всего собрано наборов данных: {metadata['total_datasets']}")
for category, count in metadata["datasets_count"].items():
    print(f"  {category}: {count} наборов данных")
