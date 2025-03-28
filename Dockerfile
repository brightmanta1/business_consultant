FROM python:3.10-slim

WORKDIR /app

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
COPY *.py .
COPY *.md .

# Создание необходимых директорий
RUN mkdir -p /app/static/js /app/static/css /app/static/img /app/model /app/config

# Копирование файлов модели и статических файлов
COPY model/ /app/model/
COPY static/ /app/static/

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создание директории для адаптеров модели
RUN mkdir -p /app/model/adapters

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["python", "web_app.py"]
