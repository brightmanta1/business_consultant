#!/usr/bin/env python3
"""
Скрипт для сбора данных о компаниях с использованием Yahoo Finance API
"""

import sys
import json
import os
from datetime import datetime


# Стало
from data_api import ApiClient  # Если data_api.py находится в той же директории


# Создаем клиент API
client = ApiClient()

# Список компаний для сбора данных
companies = [
    # Технологический сектор
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "INTC", "name": "Intel Corporation"},
    
    # Финансовый сектор
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "BAC", "name": "Bank of America Corporation"},
    {"symbol": "WFC", "name": "Wells Fargo & Company"},
    {"symbol": "GS", "name": "The Goldman Sachs Group, Inc."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "MA", "name": "Mastercard Incorporated"},
    
    # Здравоохранение
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "PFE", "name": "Pfizer Inc."},
    {"symbol": "UNH", "name": "UnitedHealth Group Incorporated"},
    {"symbol": "MRK", "name": "Merck & Co., Inc."},
    
    # Потребительский сектор
    {"symbol": "PG", "name": "The Procter & Gamble Company"},
    {"symbol": "KO", "name": "The Coca-Cola Company"},
    {"symbol": "PEP", "name": "PepsiCo, Inc."},
    {"symbol": "WMT", "name": "Walmart Inc."},
    {"symbol": "MCD", "name": "McDonald's Corporation"},
    
    # Промышленный сектор
    {"symbol": "GE", "name": "General Electric Company"},
    {"symbol": "BA", "name": "The Boeing Company"},
    {"symbol": "CAT", "name": "Caterpillar Inc."},
    {"symbol": "MMM", "name": "3M Company"},
    
    # Энергетический сектор
    {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
    {"symbol": "CVX", "name": "Chevron Corporation"},
]

# Создаем директории для хранения данных
output_dir = os.path.dirname(os.path.abspath(__file__))
profiles_dir = os.path.join(output_dir, "company_profiles")
insights_dir = os.path.join(output_dir, "company_insights")

os.makedirs(profiles_dir, exist_ok=True)
os.makedirs(insights_dir, exist_ok=True)

# Функция для сохранения данных в JSON-файл
def save_to_json(data, directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

# Сбор данных о компаниях
collected_data = {
    "profiles": [],
    "insights": []
}

for company in companies:
    symbol = company["symbol"]
    name = company["name"]
    
    print(f"Сбор данных для компании: {name} ({symbol})")
    
    try:
        # Получение профиля компании
        profile = client.call_api('YahooFinance/get_stock_profile', query={'symbol': symbol, 'region': 'US', 'lang': 'en-US'})
        profile_filename = f"{symbol}_profile.json"
        profile_path = save_to_json(profile, profiles_dir, profile_filename)
        print(f"  Профиль сохранен в: {profile_path}")
        
        # Добавляем информацию в список собранных данных
        collected_data["profiles"].append({
            "symbol": symbol,
            "name": name,
            "file": profile_filename
        })
        
        # Получение аналитических данных о компании
        insights = client.call_api('YahooFinance/get_stock_insights', query={'symbol': symbol})
        insights_filename = f"{symbol}_insights.json"
        insights_path = save_to_json(insights, insights_dir, insights_filename)
        print(f"  Аналитические данные сохранены в: {insights_path}")
        
        # Добавляем информацию в список собранных данных
        collected_data["insights"].append({
            "symbol": symbol,
            "name": name,
            "file": insights_filename
        })
        
    except Exception as e:
        print(f"  Ошибка при сборе данных для {symbol}: {str(e)}")

# Сохраняем метаданные о собранных данных
metadata = {
    "collection_date": datetime.now().isoformat(),
    "companies_count": len(companies),
    "successful_profiles": len(collected_data["profiles"]),
    "successful_insights": len(collected_data["insights"]),
    "companies": collected_data
}

metadata_path = save_to_json(metadata, output_dir, "yahoo_finance_metadata.json")
print(f"\nМетаданные сохранены в: {metadata_path}")
print(f"Всего собрано профилей компаний: {len(collected_data['profiles'])}")
print(f"Всего собрано аналитических данных: {len(collected_data['insights'])}")
