#!/usr/bin/env python3
"""
Скрипт для поиска и сбора данных из Kaggle Datasets
"""

import os
import json
import time
from datetime import datetime

# Создаем директорию для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
kaggle_dir = os.path.join(output_dir, "kaggle_datasets")
os.makedirs(kaggle_dir, exist_ok=True)

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

# Функция для установки Kaggle API
def setup_kaggle_api():
    try:
        # Проверяем, установлен ли kaggle
        import kaggle
        print("Kaggle API уже установлен")
        return True
    except ImportError:
        print("Установка Kaggle API...")
        os.system("pip install kaggle")
        try:
            import kaggle
            print("Kaggle API успешно установлен")
            return True
        except ImportError:
            print("Не удалось установить Kaggle API")
            return False

# Функция для настройки учетных данных Kaggle
def setup_kaggle_credentials():
    # Создаем директорию для учетных данных Kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Создаем файл с учетными данными
    # Обратите внимание: в реальном сценарии вам нужно будет использовать свои учетные данные
    credentials = {
        "username": "demo_user",
        "key": "demo_key"
    }
    
    credentials_path = os.path.join(kaggle_dir, "kaggle.json")
    
    # Проверяем, существует ли файл с учетными данными
    if not os.path.exists(credentials_path):
        print("Создание демонстрационного файла с учетными данными Kaggle...")
        with open(credentials_path, 'w') as f:
            json.dump(credentials, f)
        
        # Устанавливаем правильные разрешения
        os.chmod(credentials_path, 0o600)
    
    return os.path.exists(credentials_path)

# Функция для поиска наборов данных на Kaggle
def search_kaggle_datasets(query, max_datasets=5):
    try:
        import kaggle
        
        print(f"Поиск наборов данных по запросу: '{query}'")
        
        # Поиск наборов данных
        datasets = kaggle.api.dataset_list(search=query, sort_by="relevance")
        
        # Ограничиваем количество результатов
        return list(datasets)[:max_datasets]
    
    except Exception as e:
        print(f"Ошибка при поиске наборов данных: {str(e)}")
        return []

# Функция для загрузки набора данных с Kaggle
def download_kaggle_dataset(dataset_ref, download_dir):
    try:
        import kaggle
        
        print(f"Загрузка набора данных: {dataset_ref}")
        
        # Загрузка набора данных
        kaggle.api.dataset_download_files(dataset_ref, path=download_dir, unzip=True)
        
        return True
    
    except Exception as e:
        print(f"Ошибка при загрузке набора данных: {str(e)}")
        return False

# Функция для сбора метаданных о наборе данных
def get_dataset_metadata(dataset_ref):
    try:
        import kaggle
        
        # Получение метаданных о наборе данных
        dataset_info = kaggle.api.dataset_view(dataset_ref)
        
        return {
            "ref": dataset_ref,
            "title": dataset_info.title,
            "size": dataset_info.size,
            "lastUpdated": dataset_info.lastUpdated,
            "downloadCount": dataset_info.downloadCount,
            "voteCount": dataset_info.voteCount,
            "description": dataset_info.description
        }
    
    except Exception as e:
        print(f"Ошибка при получении метаданных о наборе данных: {str(e)}")
        return {
            "ref": dataset_ref,
            "error": str(e)
        }

# Основная функция для сбора данных
def collect_kaggle_data():
    # Устанавливаем Kaggle API
    if not setup_kaggle_api():
        print("Не удалось установить Kaggle API. Сбор данных из Kaggle невозможен.")
        return False
    
    # Настраиваем учетные данные Kaggle
    if not setup_kaggle_credentials():
        print("Не удалось настроить учетные данные Kaggle. Сбор данных из Kaggle невозможен.")
        return False
    
    # Сбор данных по каждой области консалтинга
    collected_data = {}
    
    for category, terms in search_terms.items():
        print(f"\nПоиск данных для категории: {category}")
        category_dir = os.path.join(kaggle_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        category_datasets = []
        
        for term in terms:
            # Поиск наборов данных
            datasets = search_kaggle_datasets(term)
            
            print(f"Найдено {len(datasets)} наборов данных для запроса '{term}'")
            
            for dataset in datasets:
                dataset_ref = f"{dataset.owner}/{dataset.name}"
                
                # Создаем директорию для набора данных
                dataset_dir = os.path.join(category_dir, dataset.name)
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Получаем метаданные о наборе данных
                metadata = get_dataset_metadata(dataset_ref)
                
                # Сохраняем метаданные
                metadata_path = save_to_json(metadata, dataset_dir, "metadata.json")
                print(f"Метаданные сохранены в: {metadata_path}")
                
                # Загружаем набор данных
                if download_kaggle_dataset(dataset_ref, dataset_dir):
                    print(f"Набор данных загружен в: {dataset_dir}")
                    
                    # Добавляем информацию в список собранных данных
                    category_datasets.append({
                        "ref": dataset_ref,
                        "name": dataset.name,
                        "title": metadata.get("title", dataset.name),
                        "dir": dataset_dir
                    })
                
                # Делаем паузу, чтобы не перегружать API
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
    
    metadata_path = save_to_json(metadata, kaggle_dir, "kaggle_metadata.json")
    print(f"\nМетаданные сохранены в: {metadata_path}")
    print(f"Всего собрано наборов данных: {metadata['total_datasets']}")
    for category, count in metadata["datasets_count"].items():
        print(f"  {category}: {count} наборов данных")
    
    return True

# Запуск сбора данных
if __name__ == "__main__":
    print("Начало сбора данных из Kaggle Datasets")
    collect_kaggle_data()
    print("Сбор данных из Kaggle Datasets завершен")
