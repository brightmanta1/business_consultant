"""
Модуль для интеграции API с FastAPI приложением.
Связывает API интерфейсы с моделью бизнес-консультанта.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# Импортируем интеграцию с моделью
from model_integration import model_integration

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Модели данных
class ConsultingRequest(BaseModel):
    """Модель для запроса консультации."""
    query: str = Field(..., description="Запрос пользователя")
    domain: str = Field(..., description="Область консалтинга (it, hr, investment, financial, operations, management)")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Дополнительные параметры для запроса")

class ConsultingResponse(BaseModel):
    """Модель для ответа на запрос консультации."""
    result: Dict[str, Any]
    recommendations: List[str]
    data_sources: List[str]
    timestamp: str

# Инициализация FastAPI
app = FastAPI(
    title="AI Business Consulting API",
    description="API для модели ИИ бизнес-консалтинга, охватывающей различные области консалтинга",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Корневой маршрут
@app.get("/")
async def root():
    return {
        "message": "AI Business Consulting API",
        "version": "1.0.0",
        "domains": [
            "financial", 
            "operations", 
            "management", 
            "it", 
            "hr", 
            "investment"
        ]
    }

# Общий маршрут для консалтинга
@app.post("/api/consulting", response_model=ConsultingResponse)
async def get_consulting(request: ConsultingRequest):
    """
    Общий эндпоинт для получения консультации в любой области.
    Перенаправляет запрос на соответствующий специализированный API.
    """
    domain = request.domain.lower()
    
    valid_domains = ["financial", "operations", "management", "it", "hr", "investment"]
    if domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Неизвестная область консалтинга: {domain}")
    
    # Преобразуем domain для соответствия ожиданиям модели
    model_domain_mapping = {
        "operations": "operations",
        "financial": "financial",
        "management": "management",
        "it": "it",
        "hr": "hr",
        "investment": "investment"
    }
    
    model_domain = model_domain_mapping.get(domain, domain)
    
    # Вызываем модель через интеграционный слой
    model_response = model_integration.process_query(
        domain=model_domain,
        query=request.query,
        context=request.parameters,
        max_length=512
    )
    
    # Преобразуем ответ модели в формат API
    from datetime import datetime
    
    # Извлекаем рекомендации из ответа модели
    recommendations = model_response.get("recommendations", [])
    if not recommendations and "response" in model_response:
        # Если рекомендаций нет, но есть текст ответа, пытаемся извлечь рекомендации из текста
        response_text = model_response["response"]
        if "рекомендуется" in response_text.lower():
            recommendations = [response_text]
    
    # Формируем список источников данных
    data_sources = model_response.get("references", ["AI Business Consultant Model"])
    
    return ConsultingResponse(
        result={
            "analysis": model_response.get("response", ""),
            "confidence": model_response.get("confidence", 0.0),
            "domain": domain,
            "query": request.query,
            "processing_time": model_response.get("processing_time", 0.0)
        },
        recommendations=recommendations,
        data_sources=data_sources,
        timestamp=datetime.now().isoformat()
    )

# Маршруты для конкретных областей консалтинга
@app.post("/api/financial", response_model=ConsultingResponse)
async def financial_consulting(request: ConsultingRequest):
    """
    API для финансового консалтинга.
    Использует данные из Yahoo Finance и DataBank для анализа финансовых показателей.
    """
    # Устанавливаем домен финансового консалтинга
    request.domain = "financial"
    return await get_consulting(request)

@app.post("/api/operations", response_model=ConsultingResponse)
async def operational_consulting(request: ConsultingRequest):
    """
    API для операционного консалтинга.
    Анализирует операционную эффективность и предлагает улучшения.
    """
    # Устанавливаем домен операционного консалтинга
    request.domain = "operations"
    return await get_consulting(request)

@app.post("/api/management", response_model=ConsultingResponse)
async def management_consulting(request: ConsultingRequest):
    """
    API для управленческого консалтинга.
    Предоставляет рекомендации по улучшению управленческих процессов.
    """
    # Устанавливаем домен управленческого консалтинга
    request.domain = "management"
    return await get_consulting(request)

@app.post("/api/it", response_model=ConsultingResponse)
async def it_consulting(request: ConsultingRequest):
    """
    API для IT-консалтинга.
    Анализирует IT-инфраструктуру и предлагает технологические решения.
    """
    # Устанавливаем домен IT-консалтинга
    request.domain = "it"
    return await get_consulting(request)

@app.post("/api/hr", response_model=ConsultingResponse)
async def hr_consulting(request: ConsultingRequest):
    """
    API для HR-консалтинга.
    Анализирует кадровые процессы и предлагает стратегии управления персоналом.
    """
    # Устанавливаем домен HR-консалтинга
    request.domain = "hr"
    return await get_consulting(request)

@app.post("/api/investment", response_model=ConsultingResponse)
async def investment_consulting(request: ConsultingRequest):
    """
    API для инвестиционного консалтинга.
    Анализирует инвестиционные возможности и предлагает стратегии.
    """
    # Устанавливаем домен инвестиционного консалтинга
    request.domain = "investment"
    return await get_consulting(request)

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
