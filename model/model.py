"""
Модуль для взаимодействия с ИИ-моделью бизнес-консультанта.
Реализует логику обработки запросов и взаимодействия с моделью.
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
import redis
from prometheus_client import Summary

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Метрики для мониторинга производительности модели
MODEL_INFERENCE_TIME = Summary('model_inference_time', 'Время инференса модели', ['domain'])
EMBEDDING_GENERATION_TIME = Summary('embedding_generation_time', 'Время генерации эмбеддингов')
CACHE_HIT_RATIO = Summary('cache_hit_ratio', 'Соотношение попаданий в кэш', ['domain'])

class BusinessConsultantModel:
    """Класс для взаимодействия с ИИ-моделью бизнес-консультанта."""
    
    def __init__(self, config: Dict):
        """Инициализация модели."""
        self.config = config
        self.model_config = config.get("model", {})
        self.cache_config = config.get("cache", {})
        
        # Инициализация кэша
        self.cache_enabled = self.cache_config.get("enabled", False)
        if self.cache_enabled:
            self.redis_client = redis.Redis(
                host=self.cache_config.get("redis_host", "localhost"),
                port=self.cache_config.get("redis_port", 6379),
                db=self.cache_config.get("redis_db", 0),
                password=self.cache_config.get("redis_password", None),
            )
            logger.info("Кэш Redis инициализирован")
        
        # Загрузка базовой модели
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.model_config.get("use_gpu", True) else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.get("tokenizer_path", "DeepPavlov/rubert-base-cased")
        )
        logger.info("Токенизатор загружен")
        
        # Загрузка базовой модели
        self.base_model = AutoModel.from_pretrained(
            self.model_config.get("base_model_path", "DeepPavlov/rubert-base-cased")
        ).to(self.device)
        logger.info("Базовая модель загружена")
        
        # Загрузка доменных адаптеров
        self.domain_adapters = {}
        self.load_domain_adapters()
        
        # Загрузка студенческой модели (если указана)
        self.student_model = None
        if self.model_config.get("use_student_model", False):
            self.load_student_model()
        
        logger.info("ИИ-модель бизнес-консультанта инициализирована")
    
    def load_domain_adapters(self):
        """Загрузка доменных адаптеров для различных областей консалтинга."""
        domains = self.model_config.get("domains", [])
        adapter_path = self.model_config.get("adapter_path", "/app/models/adapters")
        
        for domain in domains:
            try:
                adapter_file = f"{adapter_path}/{domain}_adapter.pt"
                if os.path.exists(adapter_file):
                    self.domain_adapters[domain] = torch.load(adapter_file, map_location=self.device)
                    logger.info(f"Адаптер для домена {domain} загружен")
                else:
                    logger.warning(f"Адаптер для домена {domain} не найден по пути {adapter_file}")
            except Exception as e:
                logger.error(f"Ошибка загрузки адаптера для домена {domain}: {e}")
    
    def load_student_model(self):
        """Загрузка студенческой модели."""
        try:
            student_model_path = self.model_config.get("student_model_path", "/app/models/student_model.pt")
            if os.path.exists(student_model_path):
                self.student_model = torch.load(student_model_path, map_location=self.device)
                logger.info("Студенческая модель загружена")
            else:
                logger.warning(f"Студенческая модель не найдена по пути {student_model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки студенческой модели: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга для текста."""
        start_time = time.time()
        
        # Проверка кэша
        if self.cache_enabled:
            cache_key = f"embedding:{hash(text)}"
            cached_embedding = self.redis_client.get(cache_key)
            if cached_embedding:
                return np.frombuffer(cached_embedding, dtype=np.float32)
        
        # Токенизация текста
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.get("max_token_length", 512)
        ).to(self.device)
        
        # Получение эмбеддинга
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  # [CLS] токен
        
        # Сохранение в кэш
        if self.cache_enabled:
            self.redis_client.set(
                cache_key,
                embedding.astype(np.float32).tobytes(),
                ex=self.cache_config.get("embedding_ttl", 3600)
            )
        
        EMBEDDING_GENERATION_TIME.observe(time.time() - start_time)
        return embedding
    
    def process_query(self, domain: str, query: str, context: Optional[Dict[str, Any]] = None, max_length: int = 512) -> Dict[str, Any]:
        """Обработка запроса пользователя."""
        start_time = time.time()
        
        # Проверка кэша
        cache_hit = False
        if self.cache_enabled:
            cache_key = f"query:{domain}:{hash(query)}:{hash(str(context))}"
            cached_response = self.redis_client.get(cache_key)
            if cached_response:
                cache_hit = True
                CACHE_HIT_RATIO.labels(domain=domain).observe(1.0)
                return json.loads(cached_response)
            else:
                CACHE_HIT_RATIO.labels(domain=domain).observe(0.0)
        
        # Подготовка входных данных
        inputs = {
            "query": query,
            "domain": domain,
            "context": context or {}
        }
        
        # Выбор модели для инференса
        if self.student_model and self.model_config.get("use_student_model", False):
            # Использование студенческой модели
            response = self._inference_with_student_model(inputs, max_length)
        else:
            # Использование базовой модели с адаптерами
            response = self._inference_with_base_model(inputs, max_length)
        
        # Добавление метаданных
        response["processing_time"] = time.time() - start_time
        
        # Сохранение в кэш
        if self.cache_enabled and not cache_hit:
            self.redis_client.set(
                cache_key,
                json.dumps(response),
                ex=self.cache_config.get("response_ttl", 3600)
            )
        
        MODEL_INFERENCE_TIME.labels(domain=domain).observe(response["processing_time"])
        return response
    
    def _inference_with_base_model(self, inputs: Dict[str, Any], max_length: int) -> Dict[str, Any]:
        """Инференс с использованием базовой модели и доменных адаптеров."""
        domain = inputs["domain"]
        query = inputs["query"]
        context = inputs["context"]
        
        # Проверка наличия адаптера для домена
        if domain not in self.domain_adapters:
            logger.warning(f"Адаптер для домена {domain} не найден, используется базовая модель")
        
        # Токенизация запроса
        tokenized_input = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.get("max_token_length", 512)
        ).to(self.device)
        
        # Получение эмбеддинга запроса
        with torch.no_grad():
            base_outputs = self.base_model(**tokenized_input)
            query_embedding = base_outputs.last_hidden_state
            
            # Применение доменного адаптера (если есть)
            if domain in self.domain_adapters:
                adapter = self.domain_adapters[domain]
                domain_outputs = adapter(query_embedding)
            else:
                domain_outputs = query_embedding
        
        # В реальной системе здесь будет генерация ответа на основе эмбеддингов
        # Для примера возвращаем заглушку
        
        # Имитация ответа модели
        domain_responses = {
            "it": "Для оптимизации IT-инфраструктуры вашей компании рекомендуется внедрить микросервисную архитектуру и контейнеризацию с использованием Docker и Kubernetes. Это повысит масштабируемость и отказоустойчивость системы.",
            "hr": "Для повышения эффективности процесса найма рекомендуется внедрить структурированные интервью и оценку по компетенциям. Это позволит снизить субъективность оценки и повысить качество найма.",
            "investment": "На основе анализа рыночных трендов и финансовых показателей компании, рекомендуется диверсифицировать инвестиционный портфель, увеличив долю технологического сектора и уменьшив долю сырьевых активов.",
            "financial": "Для оптимизации налогообложения рекомендуется рассмотреть возможность перехода на упрощенную систему налогообложения и использование налоговых льгот для инновационных предприятий.",
            "operations": "Для повышения операционной эффективности рекомендуется внедрить методологию Lean Six Sigma и автоматизировать рутинные процессы с использованием RPA (Robotic Process Automation).",
            "management": "Для улучшения стратегического управления рекомендуется внедрить систему сбалансированных показателей (BSC) и регулярный стратегический анализ с использованием методологии PESTEL и SWOT."
        }
        
        response_text = domain_responses.get(domain.lower(), "Консультация по вашему запросу.")
        
        return {
            "domain": domain,
            "query": query,
            "response": response_text,
            "confidence": 0.92,
            "references": [
                "https://example.com/reference1",
                "https://example.com/reference2"
            ],
            "recommendations": [
                "Рекомендация 1",
                "Рекомендация 2",
                "Рекомендация 3"
            ]
        }
    
    def _inference_with_student_model(self, inputs: Dict[str, Any], max_length: int) -> Dict[str, Any]:
        """Инференс с использованием студенческой модели."""
        if not self.student_model:
            logger.warning("Студенческая модель не загружена, используется базовая модель")
            return self._inference_with_base_model(inputs, max_length)
        
        domain = inputs["domain"]
        query = inputs["query"]
        context = inputs["context"]
        
        # Токенизация запроса
        tokenized_input = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.get("max_token_length", 512)
        ).to(self.device)
        
        # Получение ответа от студенческой модели
        with torch.no_grad():
            outputs = self.student_model(**tokenized_input)
        
        # В реальной системе здесь будет декодирование выходных данных модели
        # Для примера возвращаем заглушку
        
        # Имитация ответа модели
        domain_responses = {
            "it": "Для оптимизации IT-инфраструктуры вашей компании рекомендуется внедрить микросервисную архитектуру и контейнеризацию с использованием Docker и Kubernetes. Это повысит масштабируемость и отказоустойчивость системы.",
            "hr": "Для повышения эффективности процесса найма рекомендуется внедрить структурированные интервью и оценку по компетенциям. Это позволит снизить субъективность оценки и повысить качество найма.",
            "investment": "На основе анализа рыночных трендов и финансовых показателей компании, рекомендуется диверсифицировать инвестиционный портфель, увеличив долю технологического сектора и уменьшив долю сырьевых активов.",
            "financial": "Для оптимизации налогообложения рекомендуется рассмотреть возможность перехода на упрощенную систему налогообложения и использование налоговых льгот для инновационных предприятий.",
            "operations": "Для повышения операционной эффективности рекомендуется внедрить методологию Lean Six Sigma и автоматизировать рутинные процессы с использованием RPA (Robotic Process Automation).",
            "management": "Для улучшения стратегического управления рекомендуется внедрить систему сбалансированных показателей (BSC) и регулярный стратегический анализ с использованием методологии PESTEL и SWOT."
        }
        
        response_text = domain_responses.get(domain.lower(), "Консультация по вашему запросу.")
        
        return {
            "domain": domain,
            "query": query,
            "response": response_text,
            "confidence": 0.89,  # Студенческая модель обычно имеет немного меньшую уверенность
            "references": [
                "https://example.com/reference1",
                "https://example.com/reference2"
            ],
            "recommendations": [
                "Рекомендация 1",
                "Рекомендация 2",
                "Рекомендация 3"
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        return {
            "base_model": self.model_config.get("base_model_path", "DeepPavlov/rubert-base-cased"),
            "domains": list(self.domain_adapters.keys()),
            "has_student_model": self.student_model is not None,
            "device": str(self.device),
            "cache_enabled": self.cache_enabled,
            "max_token_length": self.model_config.get("max_token_length", 512)
        }
