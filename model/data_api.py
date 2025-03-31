#!/usr/bin/env python3
"""
Модуль для доступа к внешним API источникам данных.
Предоставляет унифицированный интерфейс для работы с различными API.
"""

import os
import json
import requests
import logging
from typing import Dict, Any, Optional, List, Union

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ApiClient:
    """Клиент для доступа к различным API источникам данных."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Инициализация клиента API.
        
        Args:
            api_key: API ключ для доступа к платным API (опционально)
            cache_dir: Директория для кэширования результатов запросов (опционально)
        """
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Конфигурация доступных API
        self.api_config = {
            "YahooFinance/get_stock_profile": {
                "url": "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                "params": {
                    "modules": "summaryProfile",
                    "region": "{region}",
                    "lang": "{lang}"
                },
                "method": "GET"
            },
            "YahooFinance/get_stock_insights": {
                "url": "https://query1.finance.yahoo.com/ws/insights/v1/finance/insights",
                "params": {
                    "symbol": "{symbol}"
                },
                "method": "GET"
            },
            "DataBank/indicator_data": {
                "url": "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
                "params": {
                    "format": "json",
                    "per_page": "100"
                },
                "method": "GET"
            },
            "DataBank/indicator_list": {
                "url": "https://api.worldbank.org/v2/indicators",
                "params": {
                    "format": "json",
                    "per_page": "{pageSize}",
                    "page": "{page}"
                },
                "method": "GET"
            },
            "DataBank/indicator_detail": {
                "url": "https://api.worldbank.org/v2/indicator/{indicatorCode}",
                "params": {
                    "format": "json"
                },
                "method": "GET"
            }
        }
        
        logger.info("API клиент инициализирован")
    
    def call_api(self, api_name: str, query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Вызов API по имени с параметрами запроса.
        
        Args:
            api_name: Имя API в формате "Provider/endpoint"
            query: Параметры запроса
            
        Returns:
            Dict: Результат запроса в формате JSON
        """
        query = query or {}
        
        # Проверка существования API
        if api_name not in self.api_config:
            raise ValueError(f"Неизвестный API: {api_name}")
        
        # Получение конфигурации API
        api_config = self.api_config[api_name]
        
        # Формирование URL и параметров запроса
        url = api_config["url"]
        params = api_config["params"].copy()
        method = api_config["method"]
        
        # Подстановка параметров в URL
        for key, value in query.items():
            if "{" + key + "}" in url:
                url = url.replace("{" + key + "}", str(value))
        
        # Подстановка параметров в query params
        for key, value in params.items():
            if isinstance(value, str) and "{" in value and "}" in value:
                param_key = value[1:-1]  # Удаляем фигурные скобки
                if param_key in query:
                    params[key] = query[param_key]
        
        # Добавление остальных параметров
        for key, value in query.items():
            if key not in params:
                params[key] = value
        
        # Проверка кэша
        cache_key = f"{api_name}_{json.dumps(query, sort_keys=True)}"
        cache_file = os.path.join(self.cache_dir, cache_key.replace("/", "_") + ".json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Данные загружены из кэша: {cache_file}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка при чтении кэша: {str(e)}")
        
        # Выполнение запроса
        try:
            logger.info(f"Выполнение запроса к API: {api_name}")
            
            # В реальном приложении здесь был бы настоящий запрос к API
            # Для примера возвращаем заглушку данных
            
            # Заглушки для разных API
            if api_name == "YahooFinance/get_stock_profile":
                response = self._mock_yahoo_profile(query.get("symbol", "AAPL"))
            elif api_name == "YahooFinance/get_stock_insights":
                response = self._mock_yahoo_insights(query.get("symbol", "AAPL"))
            elif api_name == "DataBank/indicator_data":
                response = self._mock_databank_indicator(query.get("indicator", "NY.GDP.MKTP.CD"), query.get("country", "US"))
            elif api_name == "DataBank/indicator_list":
                response = self._mock_databank_indicator_list(query.get("page", 1), query.get("pageSize", 10))
            elif api_name == "DataBank/indicator_detail":
                response = self._mock_databank_indicator_detail(query.get("indicatorCode", "NY.GDP.MKTP.CD"))
            else:
                response = {"error": "API не реализован"}
            
            # Сохранение в кэш
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса к API {api_name}: {str(e)}")
            raise
    
    def _mock_yahoo_profile(self, symbol: str) -> Dict[str, Any]:
        """Заглушка для профиля компании Yahoo Finance."""
        return {
            "quoteSummary": {
                "result": [{
                    "summaryProfile": {
                        "address1": "One Apple Park Way",
                        "city": "Cupertino",
                        "state": "CA",
                        "zip": "95014",
                        "country": "United States",
                        "phone": "408 996 1010",
                        "website": "https://www.apple.com",
                        "industry": "Consumer Electronics",
                        "sector": "Technology",
                        "longBusinessSummary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
                        "fullTimeEmployees": 164000,
                        "companyOfficers": []
                    }
                }],
                "error": None
            }
        }
    
    def _mock_yahoo_insights(self, symbol: str) -> Dict[str, Any]:
        """Заглушка для аналитических данных Yahoo Finance."""
        return {
            "finance": {
                "result": {
                    "symbol": symbol,
                    "instrumentInfo": {
                        "technicalEvents": {
                            "provider": "Trading Central",
                            "sector": "Technology",
                            "shortTermOutlook": {
                                "stateDescription": "Bullish",
                                "direction": "Bullish",
                                "score": 0.8,
                                "scoreDescription": "Strong"
                            },
                            "intermediateTermOutlook": {
                                "stateDescription": "Bullish",
                                "direction": "Bullish",
                                "score": 0.7,
                                "scoreDescription": "Strong"
                            },
                            "longTermOutlook": {
                                "stateDescription": "Neutral",
                                "direction": "Neutral",
                                "score": 0.5,
                                "scoreDescription": "Neutral"
                            }
                        },
                        "valuation": {
                            "color": 2,
                            "description": "Overvalued",
                            "discount": "-5.2%",
                            "relativeValue": "Overvalued compared to sector"
                        }
                    },
                    "recommendation": {
                        "targetPrice": 185.0,
                        "provider": "Research",
                        "rating": "Buy"
                    }
                },
                "error": None
            }
        }
    
    def _mock_databank_indicator(self, indicator: str, country: str) -> Dict[str, Any]:
        """Заглушка для данных индикатора DataBank."""
        return {
            "countryCode": country,
            "countryName": "United States",
            "indicatorCode": indicator,
            "indicatorName": "GDP (current US$)",
            "data": {
                "2018": 20544343456936.5,
                "2019": 21427675352862.7,
                "2020": 20893746000000.0,
                "2021": 22996100000000.0,
                "2022": 25462700000000.0
            }
        }
    
    def _mock_databank_indicator_list(self, page: int, page_size: int) -> Dict[str, Any]:
        """Заглушка для списка индикаторов DataBank."""
        indicators = [
            {"indicatorCode": "NY.GDP.MKTP.CD", "indicatorName": "GDP (current US$)"},
            {"indicatorCode": "NY.GDP.PCAP.CD", "indicatorName": "GDP per capita (current US$)"},
            {"indicatorCode": "FP.CPI.TOTL.ZG", "indicatorName": "Inflation, consumer prices (annual %)"},
            {"indicatorCode": "SL.UEM.TOTL.ZS", "indicatorName": "Unemployment, total (% of total labor force)"},
            {"indicatorCode": "NE.EXP.GNFS.ZS", "indicatorName": "Exports of goods and services (% of GDP)"},
            {"indicatorCode": "NE.IMP.GNFS.ZS", "indicatorName": "Imports of goods and services (% of GDP)"},
            {"indicatorCode": "BX.KLT.DINV.WD.GD.ZS", "indicatorName": "Foreign direct investment, net inflows (% of GDP)"},
            {"indicatorCode": "GC.DOD.TOTL.GD.ZS", "indicatorName": "Central government debt, total (% of GDP)"},
            {"indicatorCode": "SP.POP.TOTL", "indicatorName": "Population, total"},
            {"indicatorCode": "SP.DYN.LE00.IN", "indicatorName": "Life expectancy at birth, total (years)"}
        ]
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return {
            "total": len(indicators),
            "page": page,
            "pageSize": page_size,
            "items": indicators[start_idx:end_idx]
        }
    
    def _mock_databank_indicator_detail(self, indicator_code: str) -> Dict[str, Any]:
        """Заглушка для детальной информации об индикаторе DataBank."""
        indicators = {
            "NY.GDP.MKTP.CD": {
                "indicatorCode": "NY.GDP.MKTP.CD",
                "indicatorName": "GDP (current US$)",
                "topic": "Economic Policy & Debt: National accounts: US$ at current prices: Aggregate indicators",
                "longDescription": "GDP at purchaser's prices is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in the value of the products. It is calculated without making deductions for depreciation of fabricated assets or for depletion and degradation of natural resources. Data are in current U.S. dollars. Dollar figures for GDP are converted from domestic currencies using single year official exchange rates. For a few countries where the official exchange rate does not reflect the rate effectively applied to actual foreign exchange transactions, an alternative conversion factor is used."
            },
            "NY.GDP.PCAP.CD": {
                "indicatorCode": "NY.GDP.PCAP.CD",
                "indicatorName": "GDP per capita (current US$)",
                "topic": "Economic Policy & Debt: National accounts: US$ at current prices: Per capita",
                "longDescription": "GDP per capita is gross domestic product divided by midyear population. GDP is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in the value of the products. It is calculated without making deductions for depreciation of fabricated assets or for depletion and degradation of natural resources. Data are in current U.S. dollars."
            }
        }
        
        return indicators.get(indicator_code, {
            "indicatorCode": indicator_code,
            "indicatorName": "Unknown Indicator",
            "topic": "Unknown",
            "longDescription": "No description available for this indicator."
        })
