"""
Основной модуль API для ИИ-модели бизнес-консультанта.
Реализует REST API для взаимодействия с моделью.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, start_http_server

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
def load_config():
    """Загрузка конфигурации из файла."""
    env = os.getenv("APP_ENV", "development")
    config_path = f"/app/config/{env}.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        # Возвращаем базовую конфигурацию по умолчанию
        return {
            "app": {"name": "ai-business-consultant", "version": "1.0.0", "language": "ru"},
            "api": {
                "prefix": "/api/v1",
                "cors": {"allow_origins": ["*"], "allow_methods": ["*"], "allow_headers": ["*"]},
            },
        }

config = load_config()

# Инициализация FastAPI
app = FastAPI(
    title="ИИ-модель бизнес-консультанта API",
    description="API для взаимодействия с ИИ-моделью бизнес-консультанта, охватывающей различные области консалтинга.",
    version="1.0.0",
    docs_url=config["api"].get("docs_url", "/api/docs"),
    redoc_url=config["api"].get("redoc_url", "/api/redoc"),
    openapi_url=config["api"].get("openapi_url", "/api/openapi.json"),
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors"].get("allow_origins", ["*"]),
    allow_credentials=True,
    allow_methods=config["api"]["cors"].get("allow_methods", ["*"]),
    allow_headers=config["api"]["cors"].get("allow_headers", ["*"]),
)

# Настройка метрик Prometheus
REQUEST_COUNT = Counter(
    "business_consultant_request_count", 
    "Количество запросов к API", 
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "business_consultant_request_latency_seconds", 
    "Время обработки запросов к API",
    ["method", "endpoint"]
)

# Middleware для сбора метрик
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware для сбора метрик запросов."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path, 
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method, 
        endpoint=request.url.path
    ).observe(process_time)
    
    return response

# Middleware для обработки исключений
@app.middleware("http")
async def exception_middleware(request: Request, call_next):
    """Middleware для централизованной обработки исключений."""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Необработанное исключение: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Внутренняя ошибка сервера. Пожалуйста, попробуйте позже."},
        )

# Модели данных
class ConsultingDomain(BaseModel):
    """Модель для представления области консалтинга."""
    id: str
    name: str
    description: str

class ConsultingRequest(BaseModel):
    """Модель для запроса консультации."""
    domain: str = Field(..., description="Область консалтинга (it, hr, investment, financial, operations, management)")
    query: str = Field(..., description="Запрос пользователя")
    context: Optional[Dict[str, Any]] = Field(None, description="Дополнительный контекст для запроса")
    max_length: Optional[int] = Field(512, description="Максимальная длина ответа")

class ConsultingResponse(BaseModel):
    """Модель для ответа на запрос консультации."""
    domain: str
    query: str
    response: str
    confidence: float
    processing_time: float
    references: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Модель для ответа на запрос о состоянии сервиса."""
    status: str
    version: str
    domains: List[str]

class TokenResponse(BaseModel):
    """Модель для ответа с токеном доступа."""
    access_token: str
    token_type: str
    expires_in: int

# Настройка аутентификации
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Получение текущего пользователя по токену."""
    # В реальном приложении здесь будет проверка токена и получение пользователя
    # Для примера возвращаем фиктивного пользователя
    return {"username": "user", "is_active": True}

# Эндпоинты API
@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Получение токена доступа."""
    # В реальном приложении здесь будет проверка учетных данных
    # Для примера возвращаем фиктивный токен
    return {
        "access_token": "example_token",
        "token_type": "bearer",
        "expires_in": 3600,
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса."""
    return {
        "status": "ok",
        "version": config["app"]["version"],
        "domains": config["model"]["domains"],
    }

@app.get("/domains", response_model=List[ConsultingDomain])
async def get_domains(current_user: Dict = Depends(get_current_user)):
    """Получение списка доступных областей консалтинга."""
    domains = [
        {
            "id": "it",
            "name": "IT-консалтинг",
            "description": "Консультации по вопросам информационных технологий, цифровой трансформации, разработки ПО и IT-инфраструктуры."
        },
        {
            "id": "hr",
            "name": "HR-консалтинг",
            "description": "Консультации по вопросам управления персоналом, найма, обучения, развития и удержания сотрудников."
        },
        {
            "id": "investment",
            "name": "Инвестиционный консалтинг",
            "description": "Консультации по вопросам инвестиций, управления активами, оценки инвестиционных возможностей и рисков."
        },
        {
            "id": "financial",
            "name": "Финансовый консалтинг",
            "description": "Консультации по вопросам финансового планирования, бюджетирования, налогообложения и финансового анализа."
        },
        {
            "id": "operations",
            "name": "Операционный консалтинг",
            "description": "Консультации по вопросам оптимизации бизнес-процессов, управления цепочками поставок и операционной эффективности."
        },
        {
            "id": "management",
            "name": "Управленческий консалтинг",
            "description": "Консультации по вопросам стратегического управления, организационного развития и корпоративного управления."
        }
    ]
    return domains

@app.post("/consult", response_model=ConsultingResponse)
async def get_consultation(request: ConsultingRequest, current_user: Dict = Depends(get_current_user)):
    """Получение консультации от ИИ-модели."""
    start_time = time.time()
    
    # Проверка валидности домена
    valid_domains = [domain.lower() for domain in config["model"]["domains"]]
    if request.domain.lower() not in valid_domains:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Недопустимая область консалтинга. Допустимые значения: {', '.join(valid_domains)}"
        )
    
    # В реальном приложении здесь будет вызов модели
    # Для примера возвращаем фиктивный ответ
    processing_time = time.time() - start_time
    
    # Имитация ответа модели
    domain_responses = {
        "it": "Для оптимизации IT-инфраструктуры вашей компании рекомендуется внедрить микросервисную архитектуру и контейнеризацию с использованием Docker и Kubernetes. Это повысит масштабируемость и отказоустойчивость системы.",
        "hr": "Для повышения эффективности процесса найма рекомендуется внедрить структурированные интервью и оценку по компетенциям. Это позволит снизить субъективность оценки и повысить качество найма.",
        "investment": "На основе анализа рыночных трендов и финансовых показателей компании, рекомендуется диверсифицировать инвестиционный портфель, увеличив долю технологического сектора и уменьшив долю сырьевых активов.",
        "financial": "Для оптимизации налогообложения рекомендуется рассмотреть возможность перехода на упрощенную систему налогообложения и использование налоговых льгот для инновационных предприятий.",
        "operations": "Для повышения операционной эффективности рекомендуется внедрить методологию Lean Six Sigma и автоматизировать рутинные процессы с использованием RPA (Robotic Process Automation).",
        "management": "Для улучшения стратегического управления рекомендуется внедрить систему сбалансированных показателей (BSC) и регулярный стратегический анализ с использованием методологии PESTEL и SWOT."
    }
    
    response = domain_responses.get(request.domain.lower(), "Консультация по вашему запросу.")
    
    return {
        "domain": request.domain,
        "query": request.query,
        "response": response,
        "confidence": 0.92,
        "processing_time": processing_time,
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

@app.post("/feedback")
async def submit_feedback(
    consultation_id: str,
    rating: int = Field(..., ge=1, le=5),
    comment: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Отправка обратной связи по консультации."""
    # В реальном приложении здесь будет сохранение обратной связи
    logger.info(f"Получена обратная связь: consultation_id={consultation_id}, rating={rating}, comment={comment}")
    return {"status": "success", "message": "Спасибо за вашу обратную связь!"}

# Настройка OpenAPI
def custom_openapi():
    """Настройка схемы OpenAPI."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="ИИ-модель бизнес-консультанта API",
        version="1.0.0",
        description="API для взаимодействия с ИИ-моделью бизнес-консультанта, охватывающей различные области консалтинга: IT, HR, инвестиционный, финансовый, операционный и управленческий.",
        routes=app.routes,
    )
    
    # Добавление информации о сервере
    openapi_schema["servers"] = [
        {"url": "/", "description": "Текущий сервер"},
        {"url": "https://business-consultant.ai/api/v1", "description": "Продакшн сервер"},
    ]
    
    # Добавление информации о безопасности
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "token",
                    "scopes": {}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Запуск сервера Prometheus для метрик
def start_metrics_server():
    """Запуск сервера Prometheus для метрик."""
    if config.get("monitoring", {}).get("prometheus", {}).get("enabled", False):
        port = config.get("monitoring", {}).get("prometheus", {}).get("port", 8000)
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")

# Запуск приложения
if __name__ == "__main__":
    start_metrics_server()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", 8080)),
        reload=os.getenv("APP_ENV") == "development",
    )
