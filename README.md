## Описание
Загружает новости с CryptoPanic https://cryptopanic.com/developers/api/ каждые 30 мин. и записывает в БД. Каждые 12 часов групперует новости за это время по отдельным валютам и оценивает настроение каждой, используя модель Hugging Face, сохраняет в память усредненные значения настроений. ({'negative': 0.001733948360197246, 'neutral': 0.19919784367084503, 'positive': 0.7990681529045105})

## Установка (Windows)
1. Клонирование репозитория 

```git clone https://github.com/StemPank/nn_news_analysis.git```

2. Создание виртуального окружения

```py -m venv venv```

3. Активация виртуального окружения

```.\venv\Scripts\activate```

4. Переход в директорию fisher

```cd telegram_bot_nlp```

5. Установка зависимостей

```py -m pip install -r requirements.txt```
<!-- py -m pip freeze > requirements.txt -->

6. Запуск скрипта для демонстрации возможностей fisher

```py main.py```