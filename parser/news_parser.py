import requests
from datetime import datetime

from parser.config import CryptoPanic_API_KEY, CryptoPanic_url
from db_sentiment_app import MainDatabase
from loggings import LoggerManager

logger = LoggerManager().get_named_logger("news_analyzer")

COIN_KEYWORDS = {
    "Bitcoin": ["bitcoin", "btc", "btc/usd", "btcusdt"],
    "Ethereum": ["ethereum", "eth", "eth/usd", "ethusdt"],
    "Solana": ["solana", "sol", "sol/usdt"],
    "Ripple": ["ripple", "xrp", "xrp/usd", "xrpusdt"],
    "Dogecoin": ["dogecoin", "doge", "doge/usd", "dogeusdt"],
    "Tether": ["tether", "usdt", "usdt/usd", "usd/usdt"],
}

class CryptoPanicParser:
    def __init__(self):
        self.api_key = CryptoPanic_API_KEY
        self.base_url = "https://cryptopanic.com/api/v1/posts/"

        self.data_base = MainDatabase()

    def fetch_news(self, public=True, limit=50):
        params = {
            "auth_token": self.api_key,
            "public": str(public).lower(),
            "limit": limit
        }
        
        try:
            logger.debug(f"Отправка запроса в CryptoPanic с параметрами: {params}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            logger.info(f"Получено {len(results)} новостей от CryptoPanic")
            return results
        except requests.RequestException as e:
            logger.error(f"Ошибка запроса к CryptoPanic: Статус-код {response.status_code}; {e}")
            return []

    def extract_coin(self, text, currencies=None):
        coins_found = set()

        # 1. Поиск по тегам
        if currencies:
            coin_titles = [c.get("title", "").strip() for c in currencies if c.get("title")]
            if coin_titles:
                logger.debug(f"Монеты найдены по тегам: {coin_titles}")
                coins_found.update(coin_titles)

        # 2. Поиск по ключевым словам в тексте
        text = text.lower()
        for coin, keywords in COIN_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                logger.debug(f"Монета '{coin}' найдена по ключевым словам в тексте")
                coins_found.add(coin)

        # 3. Если ничего не найдено
        if not coins_found:
            logger.debug("Монета не найдена")
            return ["Unknown"]

        return sorted(coins_found)  # или list(coins_found), если порядок не важен

    def run(self):
        logger.info("Запуск парсера CryptoPanic...")
        news = self.fetch_news()
        logger.debug(f"Начало сохранения {len(news)} новостей в БД")
        self.data_base.cryptopanic_save_news(news, self.extract_coin)
        logger.info("Парсинг и сохранение новостей завершены")

