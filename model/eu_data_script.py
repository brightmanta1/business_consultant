#!/usr/bin/env python3
"""
Скрипт для поиска и сбора данных из EU Open Data Portal
"""

import requests
import json
import os
import time
from datetime import datetime

# Создаем директорию для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
eu_data_dir = os.path.join(output_dir, "eu_data_datasets")
os.makedirs(eu_data_dir, exist_ok=True)

# Базовый URL для API EU Open Data Portal
BASE_URL = "https://data.europa.eu/api/hub/search/datasets"

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
        "employment", 
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
def search_datasets(query, limit=10):
    params = {
        "q": query,
        "limit": limit,
        "page": 1,
        "sort": "relevance"
    }
    
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при поиске данных для запроса '{query}': {str(e)}")
        return None

# Функция для загрузки набора данных
def download_dataset(dataset_id):
    url = f"https://data.europa.eu/api/hub/search/datasets/{dataset_id}"
    
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при загрузке набора данных с ID '{dataset_id}': {str(e)}")
        return None

# Сбор данных по каждой области консалтинга
collected_data = {}

for category, terms in search_terms.items():
    print(f"\nПоиск данных для категории: {category}")
    category_dir = os.path.join(eu_data_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    category_datasets = []
    
    for term in terms:
        print(f"  Поиск по запросу: '{term}'")
        search_results = search_datasets(term)
        
        if search_results and "result" in search_results:
            results = search_results.get("result", {}).get("results", [])
            print(f"  Найдено {len(results)} наборов данных")
            
            for dataset in results[:5]:  # Берем только первые 5 результатов для каждого запроса
                dataset_id = dataset.get("id")
                dataset_title = dataset.get("title", "")
                
                if dataset_id:
                    # Создаем безопасное имя файла из заголовка
                    safe_title = "".join([c if c.isalnum() else "_" for c in dataset_title])
                    safe_title = safe_title[:50]  # Ограничиваем длину
                    
                    print(f"    Загрузка набора данных: {dataset_title}")
                    
                    # Получаем полную информацию о наборе данных
                    dataset_info = download_dataset(dataset_id)
                    
                    if dataset_info and "result" in dataset_info:
                        dataset_data = dataset_info.get("result")
                        
                        # Сохраняем информацию о наборе данных
                        filename = f"{safe_title}_{dataset_id}.json"
                        filepath = save_to_json(dataset_data, category_dir, filename)
                        print(f"    Сохранено в: {filepath}")
                        
                        # Добавляем информацию в список собранных данных
                        category_datasets.append({
                            "id": dataset_id,
                            "title": dataset_title,
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

metadata_path = save_to_json(metadata, eu_data_dir, "eu_data_metadata.json")
print(f"\nМетаданные сохранены в: {metadata_path}")
print(f"Всего собрано наборов данных: {metadata['total_datasets']}")
for category, count in metadata["datasets_count"].items():
    print(f"  {category}: {count} наборов данных")
