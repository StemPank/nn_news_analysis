import time
import threading
from sentiment_app import main_app_instance

def run_app():
    main_app_instance.start()



if __name__ == "__main__":
    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()

    while True:
        time.sleep(240)

        # Получить текущие результаты
        results = main_app_instance.get_sentiment_results()

        # Например:
        btc_sentiment = results.get('Bitcoin_average_all')
        print("BTC Sentiment:", btc_sentiment)