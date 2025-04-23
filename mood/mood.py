from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

"""
    Hugging Face, Inc. — американская компания, разрабатывающая инструменты для создания приложений с использованием машинного обучения.[3] 
    Она наиболее известна своей библиотекой Transformers, созданной для приложений обработки естественного языка, и своей платформой, 
    которая позволяет пользователям обмениваться моделями машинного обучения и наборами данных.
"""

class SentimentAnalysis:
    def __init__(self):
    # Загружаем модель и токенизатор
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    # Функция для анализа настроения
    def analyze_sentiment(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**encoded_input)

        scores = output.logits[0].numpy()
        probs = softmax(scores)

        labels = ['negative', 'neutral', 'positive']
        return {label: float(prob) for label, prob in zip(labels, probs)}

if __name__ == "__main__":
    # Пример
    text = "Bitcoin is looking strong today, might break $70k soon!"
    sentiment = SentimentAnalysis()
    
    print(sentiment.analyze_sentiment(text))