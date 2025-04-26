import os, json
from datetime import datetime, timezone
import sqlite3
from loggings import LoggerManager
from dateutil import parser as date_parser
import pytz

from parser.config import COIN_KEYWORDS

logger = LoggerManager().get_named_logger("news_analyzer")

def convert_time_to_local(utc_time_str, timezone='Asia/Irkutsk'):
    try:
        utc_time = date_parser.parse(utc_time_str)
        local_tz = pytz.timezone(timezone)
        return utc_time.astimezone(local_tz).strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        logger.warning(f"Ошибка конвертации времени: {e}")
        return utc_time_str

class MainDatabase():
    """Для SQLite безопаснее и проще — использовать соединение внутри каждого метода, особенно в многопоточном окружении."""

    _instance = None  # Храним единственный экземпляр

    # def __new__(cls):
    #     if cls._instance is None:
    #         logger.info("Создание соединения с основной БД")
    #         cls._instance = super().__new__(cls) # Создаём объект один раз
    #         db_path = ('cryptonews.db')
    #         cls._instance.connection = sqlite3.connect(db_path) # Открываем одно соединение
    #         cls._instance.cursor = cls._instance.connection.cursor()
    #         cls._instance.connection.execute("PRAGMA foreign_keys = ON")  # Включаем поддержку внешних ключей

    #     return cls._instance  # Возвращаем тот же объект

    def __init__(self):
        # if not hasattr(self, "initialized"):  # Чтобы __init__ не вызывался повторно
        self.db_path = 'cryptonews.db'
        self.create_table_news()
            
            # self.initialized = True  # Пометка, что init уже был выполнен
 
    def get_connection(self):
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
    
    def create_table_news(self):
        """
        
        """
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    url TEXT UNIQUE,
                    published_at TEXT,
                    currency TEXT,
                    summary TEXT,
                    content TEXT
                )
            ''')
        
        connection.commit()
        connection.close()

    
    def cryptopanic_save_news(self, news_items, extract_coin_func):
        """
            Добавление новости из CryptoPanic
        """
        connection = self.get_connection()
        cursor = connection.cursor()
        news_items = news_items[::-1]
        for post in news_items:
            title = post.get("title", "")
            url = post.get("url", "")

            # Преобразуем время в UTC
            try:
                published_dt = date_parser.parse(post.get("published_at", ""))
                published_utc = published_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logger.warning(f"Ошибка при парсинге времени: {e}")
                published_utc = post.get("published_at", "")  # fallback
            summary = post.get("summary", "")

            # Определяем валюты
            currency_list = extract_coin_func(title + " " + summary, post.get("currencies"))
            currency_str = ", ".join(currency_list)  # Преобразуем список в строку
            content = ""

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO news (title, url, published_at, currency, summary, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (title, url, published_utc, currency_str, summary, content))
                logger.info(f"✅ Сохранена новость: [{currency_str}] {title}")
            except sqlite3.Error as e:
                logger.error(f"Ошибка записи в БД: {e}")
        
        connection.commit()
        connection.close()


    def get_news_by_currency(self, coin_keywords=COIN_KEYWORDS):
        
        connection = self.get_connection()
        cursor = connection.cursor()

        results={}
        # Поочередно для каждой монеты в COIN_KEYWORDS
        for coin, keywords in coin_keywords.items():
            # Строим строку для поиска по валюте
            currency_search = "|".join(keywords)  # Это будет искать по ключевым словам монеты

            # Запрос для получения заголовков за последние 12 часов для каждой монеты
            query = f'''
                SELECT title, currency
                FROM news
                WHERE published_at >= datetime('now', '-12 hours')
                AND currency LIKE '%{coin}%'  -- Проверяем, если в поле currency есть монета
                ORDER BY published_at DESC;
            '''
            cursor.execute(query)
            results[coin] = cursor.fetchall()
            if not results[coin]:
                logger.debug(f"Нет новостей для {coin} за последние 12 часов.")
        
        connection.close()
        return results

if __name__ == "__main__":
    database = MainDatabase()
    grouped_news = database.get_news_by_currency()
    for currency, news_items in grouped_news.items():
        print(currency, news_items)