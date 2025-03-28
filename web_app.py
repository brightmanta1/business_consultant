"""
Модуль для обслуживания статических файлов и интеграции веб-интерфейса с API.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Импортируем API приложение
from api_app import app as api_app

# Создаем основное приложение
app = FastAPI(
    title="ИИ Бизнес-Консультант",
    description="Веб-интерфейс для модели ИИ бизнес-консалтинга",
    version="1.0.0"
)

# Определяем пути к статическим файлам
static_dir = os.path.join(os.path.dirname(__file__), "static")
js_dir = os.path.join(static_dir, "js")
css_dir = os.path.join(static_dir, "css")
img_dir = os.path.join(static_dir, "img")

# Создаем директории, если они не существуют
os.makedirs(js_dir, exist_ok=True)
os.makedirs(css_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Монтируем API как подприложение
app.mount("/api", api_app)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/js", StaticFiles(directory=js_dir), name="js")
app.mount("/css", StaticFiles(directory=css_dir), name="css")
app.mount("/img", StaticFiles(directory=img_dir), name="img")

# Обработчик корневого маршрута
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Отображение главной страницы."""
    try:
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Ошибка при чтении index.html: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при загрузке страницы")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
