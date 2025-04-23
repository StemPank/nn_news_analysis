import time
import threading
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import defaultdict
import numpy as np

from parser.news_parser import CryptoPanicParser
from mood.mood import SentimentAnalysis
from db import MainDatabase
from loggings import LoggerManager

logger = LoggerManager().get_main_logger()

# Класс для основного приложения
class MainApp:
    def __init__(self):
        self.parser = CryptoPanicParser()
        # self.data_base = MainDatabase()
        self.sentiment_analysis = SentimentAnalysis()
        self.scheduler = BackgroundScheduler()

        self.sentiment_results = {}

    def fetch_and_store_news(self):
        """Метод для получения новостей с CryptoPanic каждые 30 минут."""
        logger.info(f"Запрос новостей с CryptoPanic в {datetime.now()}")
        self.parser.run()  # Получаем и сохраняем новости в базе данных

    def group_and_analyze_news(self):
        """Метод для группировки новостей и анализа настроений каждые 12 часов."""
        logger.info(f"Группировка новостей и анализ настроений в {datetime.now()}")
        data_base = MainDatabase()
        grouped_news = data_base.get_news_by_currency()
        
        self.sentiment_results = defaultdict(list) 
        self.sentiment_results.clear()  # Очищаем старые результаты

        sentiment_aggregates = defaultdict(list)  # Для сбора вероятностей

        for currency, news_items in grouped_news.items():
            logger.info(f"Обрабатываем новости для {currency}")
            for title, currency_tags in news_items:
                # logger.debug(f"Анализ настроений text:{title}")
                sentiment = self.sentiment_analysis.analyze_sentiment(title) # Анализ настроений
                self.sentiment_results[currency].append((title, sentiment))

                sentiment_aggregates[currency].append([
                    sentiment['negative'],
                    sentiment['neutral'],
                    sentiment['positive']
                ])

                logger.info(f"Заголовок: {title}")
                logger.info(f"Настроение: {sentiment}")
         # Усреднение
        for currency, values in sentiment_aggregates.items():
            average = np.mean(values, axis=0)
            avg_sentiment = {
                'negative': round(average[0], 3),
                'neutral': round(average[1], 3),
                'positive': round(average[2], 3)
            }
            logger.info(f"📊 Среднее настроение по {currency}: {avg_sentiment}")
            self.sentiment_results[currency + "_average"] = avg_sentiment

    def start(self):
        """Запуск программы."""
        # Запрашиваем новости каждые 30 минут (сразу и потом каждые 30 мин)
        self.scheduler.add_job(
            self.fetch_and_store_news,
            'interval',
            minutes=30,
            next_run_time=datetime.now()
        )

        # Группируем новости и анализируем их каждые 12 часов (тоже сразу)
        self.scheduler.add_job(
            self.group_and_analyze_news,
            'interval',
            hours=12,
            next_run_time=datetime.now()
        )

        # Стартуем планировщик
        self.scheduler.start()

        # Держим процесс в живых
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.scheduler.shutdown()

# Запуск приложения
if __name__ == "__main__":
    app = MainApp()
    app.start()