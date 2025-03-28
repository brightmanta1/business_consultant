#!/usr/bin/env python3
"""
Скрипт для сбора данных из Data USA API
"""

import requests
import json
import os
import time
from datetime import datetime

# Создаем директорию для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
data_usa_dir = os.path.join(output_dir, "data_usa_datasets")
os.makedirs(data_usa_dir, exist_ok=True)

# Базовый URL для API Data USA
BASE_URL = "https://datausa.io/api"

# Категории данных для сбора
data_categories = {
    "economy": [
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "GDP"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "GDP per Capita"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Household Income"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Household Income by Race"}}
    ],
    "employment": [
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Workforce"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Unemployment Rate"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Wage by Industry"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Wage by Occupation"}}
    ],
    "education": [
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Educational Attainment"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "University"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Degrees Awarded"}}
    ],
    "demographics": [
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Population"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Age"}},
        {"endpoint": "/data", "params": {"drilldowns": "Nation", "measures": "Race"}}
    ],
    "industry": [
        {"endpoint": "/data", "params": {"drilldowns": "Industry Group", "measures": "Establishments"}},
        {"endpoint": "/data", "params": {"drilldowns": "Industry Group", "measures": "Average Wage"}},
        {"endpoint": "/data", "params": {"drilldowns": "Industry Group", "measures": "Workers"}}
    ],
    "occupation": [
        {"endpoint": "/data", "params": {"drilldowns": "Occupation", "measures": "Workers"}},
        {"endpoint": "/data", "params": {"drilldowns": "Occupation", "measures": "Average Wage"}},
        {"endpoint": "/data", "params": {"drilldowns": "Occupation", "measures": "Average Age"}}
    ]
}

# Функция для сохранения данных в JSON-файл
def save_to_json(data, directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

# Функция для запроса данных из API
def fetch_data(endpoint, params):
    url = BASE_URL + endpoint
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при запросе данных: {str(e)}")
        return None

# Сбор данных по каждой категории
collected_data = {}

for category, queries in data_categories.items():
    print(f"\nСбор данных для категории: {category}")
    category_dir = os.path.join(data_usa_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    category_datasets = []
    
    for query in queries:
        endpoint = query["endpoint"]
        params = query["params"]
        
        # Создаем имя файла на основе параметров запроса
        filename_parts = []
        for key, value in params.items():
            filename_parts.append(f"{key}_{value}")
        filename = "_".join(filename_parts).replace(" ", "_") + ".json"
        
        print(f"  Запрос данных: {endpoint} с параметрами {params}")
        data = fetch_data(endpoint, params)
        
        if data:
            filepath = save_to_json(data, category_dir, filename)
            print(f"  Данные сохранены в: {filepath}")
            
            # Добавляем информацию в список собранных данных
            category_datasets.append({
                "endpoint": endpoint,
                "params": params,
                "file": filename
            })
        
        # Делаем паузу между запросами
        time.sleep(1)
    
    # Сохраняем информацию о собранных данных для категории
    collected_data[category] = category_datasets

# Сохраняем метаданные о собранных данных
metadata = {
    "collection_date": datetime.now().isoformat(),
    "categories": list(data_categories.keys()),
    "datasets_count": {category: len(datasets) for category, datasets in collected_data.items()},
    "total_datasets": sum(len(datasets) for datasets in collected_data.values()),
    "data": collected_data
}

metadata_path = save_to_json(metadata, data_usa_dir, "data_usa_metadata.json")
print(f"\nМетаданные сохранены в: {metadata_path}")
print(f"Всего собрано наборов данных: {metadata['total_datasets']}")
for category, count in metadata["datasets_count"].items():
    print(f"  {category}: {count} наборов данных")
