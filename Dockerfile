# Указываем базовый образ Python
FROM python:3.11

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем зависимости, необходимые для сборки numpy и libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1-mesa-glx

# Копируем main.py внутрь контейнера
COPY ./main.py /app/main.py

# Копируем requirements.txt и устанавливаем зависимости
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируйте ваше FastAPI-приложение внутрь контейнера
COPY ./ /app

# Укажите команду для запуска вашего приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]












