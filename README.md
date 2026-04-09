# arXiv Article Classifier

**Классификация научных статей из arXiv по 7 научным областям:**
- Computer Science
- Physics
- Mathematics
- Biology
- Statistics
- Electrical Engineering
- Economics

Модель определяет тематику статьи по **названию** и/или **абстракту**, показывая **топ-95% вероятностей** с визуальным прогресс-баром.

Попробовать сервис можно здесь:
**https://arxivclassifier-parhqy2zewactswnr7ixwr.streamlit.app/#ar-xiv-article-classifier**

(Первый раз может загружаться несколько минут, т.к. нужно скачать модель и веса, они хранятся по адресу: https://huggingface.co/adpokh/arxiv-model)

## Обучение модели 
(train_model_1.py)

**Модель:** `distilbert-base-uncased`

**Датасет:** [TimSchopf/arxiv_categories](https://huggingface.co/datasets/TimSchopf/arxiv_categories)

**Классы:** Physics, Mathematics, Computer Science, Statistics, Electrical Engineering, Biology, Economics
(Классы были переименованы в вышеуказанные по описанию из файла, который был приложен к датасетам по ссылке выше, а также классы Finance и Economics были объединены в один, т.к. они маленькие и похожие - файл filter_data.py)

**Размер выборки:** 30 000 статей (train) (ототбраны из оригинального датасета случайно, чтобы не так долго обучалось)

**Параметры:**
- Максимальная длина текста: 128 токенов
- Размер батча: 32
- Оптимизатор: AdamW (lr=2e-5)
- Эпохи: 3

**Результаты:**
- Accuracy (на датасете val, который использовался, как тестовый датасет): 92.64%
- Loss: 0.1261
