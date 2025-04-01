"""
Модуль конфигурации для бизнес-консультанта.
Содержит настройки для различных компонентов системы.
"""

import os
from typing import Dict, List, Any, Optional

# Базовая конфигурация
BASE_CONFIG = {
    "app": {
        "name": "AI Business Consultant",
        "version": "2.0.0",
        "description": "ИИ бизнес-консультант с поддержкой различных областей консалтинга",
        "default_language": "ru",
        "supported_languages": ["ru", "en"],
    },
    "model": {
        # Базовая многоязычная модель вместо только русскоязычной
        "base_model_path": "microsoft/mdeberta-v3-base",
        "tokenizer_path": "microsoft/mdeberta-v3-base",
        "use_gpu": True,
        "max_token_length": 512,
        "use_student_model": False,
        "domains": ["it", "hr", "investment", "financial", "operations", "management"],
        "adapter_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "adapters"),
        "student_model_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "student"),
    },
    "cache": {
        "enabled": True,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "redis_password": None,
        "embedding_ttl": 3600,  # 1 час
        "response_ttl": 3600,   # 1 час
    },
    "data": {
        "datasets_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "datasets"),
        "processed_data_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed"),
        "triplets_path": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "triplets"),
        "automl": {
            "enabled": True,
            "optimization_metric": "accuracy",
            "max_trials": 10,
            "timeout": 3600,  # 1 час
        },
        "maml": {
            "enabled": True,
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "num_inner_steps": 5,
        },
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        "cors_methods": ["*"],
        "cors_headers": ["*"],
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "static_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "static"),
        "templates_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "templates"),
        "localization_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "localization"),
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "app.log"),
    },
    "metrics": {
        "enabled": True,
        "prometheus_port": 9090,
    },
}

def get_config() -> Dict[str, Any]:
    """
    Получение конфигурации приложения.
    
    Returns:
        Dict[str, Any]: Конфигурация приложения
    """
    # Здесь можно добавить логику для загрузки конфигурации из файла или переменных окружения
    return BASE_CONFIG

def get_model_config() -> Dict[str, Any]:
    """
    Получение конфигурации модели.
    
    Returns:
        Dict[str, Any]: Конфигурация модели
    """
    return get_config()["model"]

def get_data_config() -> Dict[str, Any]:
    """
    Получение конфигурации для обработки данных.
    
    Returns:
        Dict[str, Any]: Конфигурация для обработки данных
    """
    return get_config()["data"]

def get_api_config() -> Dict[str, Any]:
    """
    Получение конфигурации API.
    
    Returns:
        Dict[str, Any]: Конфигурация API
    """
    return get_config()["api"]

def get_web_config() -> Dict[str, Any]:
    """
    Получение конфигурации веб-интерфейса.
    
    Returns:
        Dict[str, Any]: Конфигурация веб-интерфейса
    """
    return get_config()["web"]

def get_logging_config() -> Dict[str, Any]:
    """
    Получение конфигурации логирования.
    
    Returns:
        Dict[str, Any]: Конфигурация логирования
    """
    return get_config()["logging"]

def get_metrics_config() -> Dict[str, Any]:
    """
    Получение конфигурации метрик.
    
    Returns:
        Dict[str, Any]: Конфигурация метрик
    """
    return get_config()["metrics"]

def get_cache_config() -> Dict[str, Any]:
    """
    Получение конфигурации кэша.
    
    Returns:
        Dict[str, Any]: Конфигурация кэша
    """
    return get_config()["cache"]

def get_app_config() -> Dict[str, Any]:
    """
    Получение общей конфигурации приложения.
    
    Returns:
        Dict[str, Any]: Общая конфигурация приложения
    """
    return get_config()["app"]
