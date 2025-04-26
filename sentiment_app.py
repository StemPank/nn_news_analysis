import time
import threading
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import defaultdict
import numpy as np

from parser.news_parser import CryptoPanicParser
from mood.mood import SentimentAnalysis
from db_sentiment_app import MainDatabase
from loggings import LoggerManager

logger = LoggerManager().get_named_logger("news_analyzer")
logger_res = LoggerManager().get_named_logger("news_analyzer_results")

# Класс для основного приложения
class MainApp:
    """
        Основной класс приложения для получения новостей, анализа настроений и хранения результатов.

        Получение результатов:
            [currency + "_average_all"] - среднее настроение по всем новостям
            [currency + "_average_strong"] - среднее настроение по новостям с вероятностью > 0.7
            [currency + "_count_stats"] - статистика по количеству новостей (положительные, нейтральные и т.д.)
    """
    def __init__(self):
        self.parser = CryptoPanicParser()
        self.sentiment_analysis = SentimentAnalysis()
        self.scheduler = BackgroundScheduler()

        self.sentiment_results = defaultdict(list) 
        self.results_lock = threading.Lock()  # Добавили блокировку

    def fetch_and_store_news(self):
        """Метод для получения новостей с CryptoPanic каждые 30 минут."""
        logger.info(f"Запрос новостей с CryptoPanic в {datetime.now()}")
        self.parser.run()  # Получаем и сохраняем новости в базе данных

    def group_and_analyze_news(self):
        """Метод для группировки новостей и анализа настроений каждые 12 часов."""
        logger.info(f"Группировка новостей и анализ настроений в {datetime.now()}")
        data_base = MainDatabase()
        grouped_news = data_base.get_news_by_currency()
        
        # self.sentiment_results.clear()  # Очищаем старые результаты
        sentiment_aggregates = defaultdict(list)  # Для сбора вероятностей Для всех новостей
        threshold_aggregates = defaultdict(list)  # Только те, у кого одна из вероятностей > 0.7
        sentiment_counts_by_currency = {}  # Статистика по категориям
    
        temp_results = defaultdict(list)

        for currency, news_items in grouped_news.items():
            logger.info(f"Обрабатываем новости для {currency}")
            sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0, 'undefined': 0}

            for title, currency_tags in news_items:
                sentiment = self.sentiment_analysis.analyze_sentiment(title) # Анализ настроений
                self.sentiment_results[currency].append((title, sentiment))

                sentiment_vector = [
                    sentiment['negative'],
                    sentiment['neutral'],
                    sentiment['positive']
                ]

                sentiment_aggregates[currency].append(sentiment_vector)
                max_value = max(sentiment_vector)

                # Анализируем — если одна из вероятностей выше 0.7, засчитываем как уверенное настроение
                if max_value > 0.7:
                    max_index = sentiment_vector.index(max_value)
                    sentiment_label = ['negative', 'neutral', 'positive'][max_index]
                    sentiment_counts[sentiment_label] += 1
                    threshold_aggregates[currency].append(sentiment_vector)
                else:
                    sentiment_counts['undefined'] += 1

                logger.info(f"Заголовок: {title}")
                logger.info(f"Настроение: {sentiment}")
            
            sentiment_counts_by_currency[currency] = sentiment_counts

        with self.results_lock:
            self.sentiment_results.clear() # Очищаем старые результаты
            self.sentiment_results.update(temp_results)
            # Усреднение по всем
            for currency, values in sentiment_aggregates.items():
                average = np.mean(values, axis=0)
                avg_sentiment = {
                    'negative': round(average[0], 3),
                    'neutral': round(average[1], 3),
                    'positive': round(average[2], 3)
                }
                logger.info(f"📊 Среднее настроение по {currency} (все): {avg_sentiment}")
                logger_res.debug(f"📊 Среднее настроение по {currency} (все): {avg_sentiment}")
                self.sentiment_results[currency + "_average_all"] = avg_sentiment

            # Усреднение по выборке с > 0.7
            for currency, values in threshold_aggregates.items():
                if values:
                    average = np.mean(values, axis=0)
                    avg_filtered_sentiment = {
                        'negative': round(average[0], 3),
                        'neutral': round(average[1], 3),
                        'positive': round(average[2], 3)
                    }
                    logger.info(f"📊 Среднее настроение по {currency} (только > 0.7): {avg_filtered_sentiment}")
                    logger_res.debug(f"📊 Среднее настроение по {currency} (только > 0.7): {avg_filtered_sentiment}")
                    self.sentiment_results[currency + "_average_strong"] = avg_filtered_sentiment

            # Выводим статистику
            for currency, counts in sentiment_counts_by_currency.items():
                logger.info(f"📈 Кол-во новостей по {currency}: {counts}")
                logger_res.debug(f"📈 Кол-во новостей по {currency}: {counts}")
                self.sentiment_results[currency + "_count_stats"] = counts
            logger_res.debug(f"\n")

    def get_sentiment_results(self):
        with self.results_lock:
            return dict(self.sentiment_results)

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
            # hours=12,
            minutes=31,
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

# Синглтон или экземпляр, который можно импортировать
main_app_instance = MainApp()

# Запуск приложения
if __name__ == "__main__":
    app = MainApp()
    app.start()