import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import torch
import os
from transformers import BertModel, BertTokenizer


# Пути к директориям
TRIPLETS_DIR = '/home/ubuntu/ai_business_consultant/triplet_improvement/triplets'
RESULTS_DIR = '/home/ubuntu/ai_business_consultant/triplet_improvement/evaluation'

# Создаем директорию для результатов, если она не существует
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'visualizations'), exist_ok=True)

# Загрузка предобученной модели BERT для создания эмбеддингов
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Устанавливаем модель в режим оценки

# Функция для создания эмбеддингов с помощью BERT
def create_bert_embeddings(texts, max_length=128):
    embeddings = []
    
    for text in texts:
        # Токенизация текста
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                          padding="max_length", truncation=True)
        
        # Получение эмбеддингов
        with torch.no_grad():
            outputs = model(**inputs)
            # Используем [CLS] токен как представление всего текста
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            embeddings.append(embedding)
    
    return np.array(embeddings)

# Функция для загрузки триплетов из файла
def load_triplets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    return triplets

# Функция для оценки разнообразия триплетов
def evaluate_triplet_diversity(triplets):
    # Группируем триплеты по типам и доменам
    triplet_types = {}
    triplet_domains = {}
    
    for triplet in triplets:
        triplet_type = triplet.get('type')
        if triplet_type not in triplet_types:
            triplet_types[triplet_type] = []
        triplet_types[triplet_type].append(triplet)
        
        domain = triplet.get('domain')
        if domain not in triplet_domains:
            triplet_domains[domain] = []
        triplet_domains[domain].append(triplet)
    
    # Оцениваем разнообразие по типам
    type_diversity = len(triplet_types)
    type_distribution = {t: len(triplets) for t, triplets in triplet_types.items()}
    
    # Оцениваем разнообразие по доменам
    domain_diversity = len(triplet_domains)
    domain_distribution = {d: len(triplets) for d, triplets in triplet_domains.items()}
    
    # Визуализируем распределение триплетов по типам
    plt.figure(figsize=(12, 6))
    plt.bar(type_distribution.keys(), type_distribution.values())
    plt.title('Распределение триплетов по типам')
    plt.xlabel('Тип триплета')
    plt.ylabel('Количество триплетов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'triplet_type_distribution.png'))
    
    # Визуализируем распределение триплетов по доменам
    plt.figure(figsize=(12, 6))
    plt.bar(domain_distribution.keys(), domain_distribution.values())
    plt.title('Распределение триплетов по доменам')
    plt.xlabel('Домен')
    plt.ylabel('Количество триплетов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'triplet_domain_distribution.png'))
    
    return {
        'type_diversity': type_diversity,
        'type_distribution': type_distribution,
        'domain_diversity': domain_diversity,
        'domain_distribution': domain_distribution
    }

# Функция для оценки качества триплетов с помощью эмбеддингов
def evaluate_triplet_quality(triplets, sample_size=500):
    # Если триплетов больше sample_size, выбираем случайную выборку
    if len(triplets) > sample_size:
        sampled_triplets = random.sample(triplets, sample_size)
    else:
        sampled_triplets = triplets
    
    # Извлекаем якоря, позитивные и негативные примеры
    anchors = [t['anchor'] for t in sampled_triplets]
    positives = [t['positive'] for t in sampled_triplets]
    negatives = [t['negative'] for t in sampled_triplets]
    
    # Создаем эмбеддинги
    print("Создание эмбеддингов для якорей...")
    anchor_embeddings = create_bert_embeddings(anchors)
    
    print("Создание эмбеддингов для позитивных примеров...")
    positive_embeddings = create_bert_embeddings(positives)
    
    print("Создание эмбеддингов для негативных примеров...")
    negative_embeddings = create_bert_embeddings(negatives)
    
    # Вычисляем косинусное сходство между якорями и позитивными примерами
    pos_similarities = []
    for i in range(len(anchor_embeddings)):
        sim = cosine_similarity([anchor_embeddings[i]], [positive_embeddings[i]])[0][0]
        pos_similarities.append(sim)
    
    # Вычисляем косинусное сходство между якорями и негативными примерами
    neg_similarities = []
    for i in range(len(anchor_embeddings)):
        sim = cosine_similarity([anchor_embeddings[i]], [negative_embeddings[i]])[0][0]
        neg_similarities.append(sim)
    
    # Вычисляем разницу между сходством с позитивными и негативными примерами
    similarity_differences = [pos - neg for pos, neg in zip(pos_similarities, neg_similarities)]
    
    # Вычисляем метрики качества
    avg_pos_similarity = np.mean(pos_similarities)
    avg_neg_similarity = np.mean(neg_similarities)
    avg_similarity_diff = np.mean(similarity_differences)
    
    # Визуализируем распределение сходства
    plt.figure(figsize=(12, 6))
    plt.hist(pos_similarities, alpha=0.5, label='Позитивные примеры')
    plt.hist(neg_similarities, alpha=0.5, label='Негативные примеры')
    plt.title('Распределение косинусного сходства')
    plt.xlabel('Косинусное сходство')
    plt.ylabel('Количество триплетов')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'similarity_distribution.png'))
    
    # Визуализируем разницу в сходстве
    plt.figure(figsize=(12, 6))
    plt.hist(similarity_differences)
    plt.title('Распределение разницы в сходстве (позитивные - негативные)')
    plt.xlabel('Разница в сходстве')
    plt.ylabel('Количество триплетов')
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'similarity_difference_distribution.png'))
    
    # Визуализируем эмбеддинги с помощью t-SNE
    print("Применение t-SNE для визуализации эмбеддингов...")
    all_embeddings = np.vstack([anchor_embeddings, positive_embeddings, negative_embeddings])
    labels = ['anchor'] * len(anchor_embeddings) + ['positive'] * len(positive_embeddings) + ['negative'] * len(negative_embeddings)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Создаем DataFrame для удобства визуализации
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'type': labels
    })
    
    # Визуализируем эмбеддинги
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='type', palette='viridis')
    plt.title('t-SNE визуализация эмбеддингов')
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'embeddings_tsne.png'))
    
    return {
        'avg_pos_similarity': avg_pos_similarity,
        'avg_neg_similarity': avg_neg_similarity,
        'avg_similarity_diff': avg_similarity_diff,
        'pos_similarities': pos_similarities,
        'neg_similarities': neg_similarities,
        'similarity_differences': similarity_differences
    }

# Функция для оценки эффективности триплетов по типам
def evaluate_triplet_effectiveness_by_type(triplets):
    # Группируем триплеты по типам
    triplet_types = {}
    for triplet in triplets:
        triplet_type = triplet.get('type')
        if triplet_type not in triplet_types:
            triplet_types[triplet_type] = []
        triplet_types[triplet_type].append(triplet)
    
    # Оцениваем качество триплетов для каждого типа
    type_quality = {}
    for triplet_type, type_triplets in triplet_types.items():
        print(f"Оценка качества триплетов типа {triplet_type}...")
        # Ограничиваем количество триплетов для оценки
        sample_size = min(100, len(type_triplets))
        sampled_triplets = random.sample(type_triplets, sample_size)
        
        # Извлекаем якоря, позитивные и негативные примеры
        anchors = [t['anchor'] for t in sampled_triplets]
        positives = [t['positive'] for t in sampled_triplets]
        negatives = [t['negative'] for t in sampled_triplets]
        
        # Создаем эмбеддинги
        anchor_embeddings = create_bert_embeddings(anchors)
        positive_embeddings = create_bert_embeddings(positives)
        negative_embeddings = create_bert_embeddings(negatives)
        
        # Вычисляем косинусное сходство
        pos_similarities = []
        neg_similarities = []
        for i in range(len(anchor_embeddings)):
            pos_sim = cosine_similarity([anchor_embeddings[i]], [positive_embeddings[i]])[0][0]
            neg_sim = cosine_similarity([anchor_embeddings[i]], [negative_embeddings[i]])[0][0]
            pos_similarities.append(pos_sim)
            neg_similarities.append(neg_sim)
        
        # Вычисляем метрики качества
        avg_pos_similarity = np.mean(pos_similarities)
        avg_neg_similarity = np.mean(neg_similarities)
        avg_similarity_diff = np.mean([pos - neg for pos, neg in zip(pos_similarities, neg_similarities)])
        
        type_quality[triplet_type] = {
            'avg_pos_similarity': avg_pos_similarity,
            'avg_neg_similarity': avg_neg_similarity,
            'avg_similarity_diff': avg_similarity_diff
        }
    
    # Визуализируем результаты
    plt.figure(figsize=(15, 8))
    
    # Подготавливаем данные для графика
    types = list(type_quality.keys())
    pos_similarities = [type_quality[t]['avg_pos_similarity'] for t in types]
    neg_similarities = [type_quality[t]['avg_neg_similarity'] for t in types]
    similarity_diffs = [type_quality[t]['avg_similarity_diff'] for t in types]
    
    # Создаем группированный бар-график
    x = np.arange(len(types))
    width = 0.25
    
    plt.bar(x - width, pos_similarities, width, label='Позитивное сходство')
    plt.bar(x, neg_similarities, width, label='Негативное сходство')
    plt.bar(x + width, similarity_diffs, width, label='Разница')
    
    plt.xlabel('Тип триплета')
    plt.ylabel('Среднее косинусное сходство')
    plt.title('Качество триплетов по типам')
    plt.xticks(x, types, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'triplet_quality_by_type.png'))
    
    return type_quality

# Функция для сравнения с предыдущей версией триплетов
def compare_with_previous_triplets(new_triplets, previous_triplets_file):
    # Загружаем предыдущие триплеты, если файл существует
    if os.path.exists(previous_triplets_file):
        previous_triplets = load_triplets(previous_triplets_file)
    else:
        print(f"Файл с предыдущими триплетами {previous_triplets_file} не найден")
        return None
    
    # Оцениваем разнообразие
    new_diversity = evaluate_triplet_diversity(new_triplets)
    prev_diversity = evaluate_triplet_diversity(previous_triplets)
    
    # Оцениваем качество (на выборке)
    sample_size = min(200, min(len(new_triplets), len(previous_triplets)))
    new_sampled = random.sample(new_triplets, sample_size)
    prev_sampled = random.sample(previous_triplets, sample_size)
    
    new_quality = evaluate_triplet_quality(new_sampled, sample_size)
    prev_quality = evaluate_triplet_quality(prev_sampled, sample_size)
    
    # Сравниваем результаты
    comparison = {
        'diversity_comparison': {
            'new_type_diversity': new_diversity['type_diversity'],
            'prev_type_diversity': prev_diversity['type_diversity'],
            'new_domain_diversity': new_diversity['domain_diversity'],
            'prev_domain_diversity': prev_diversity['domain_diversity']
        },
        'quality_comparison': {
            'new_avg_pos_similarity': new_quality['avg_pos_similarity'],
            'prev_avg_pos_similarity': prev_quality['avg_pos_similarity'],
            'new_avg_neg_similarity': new_quality['avg_neg_similarity'],
            'prev_avg_neg_similarity': prev_quality['avg_neg_similarity'],
            'new_avg_similarity_diff': new_quality['avg_similarity_diff'],
            'prev_avg_similarity_diff': prev_quality['avg_similarity_diff']
        }
    }
    
    # Визуализируем сравнение
    plt.figure(figsize=(12, 6))
    
    # Сравнение разнообразия
    plt.subplot(1, 2, 1)
    diversity_data = {
        'Типы триплетов': [new_diversity['type_diversity'], prev_diversity['type_diversity']],
        'Домены': [new_diversity['domain_diversity'], prev_diversity['domain_diversity']]
    }
    
    x = np.arange(len(diversity_data))
    width = 0.35
    
    plt.bar(x - width/2, [diversity_data[k][0] for k in diversity_data], width, label='Новые триплеты')
    plt.bar(x + width/2, [diversity_data[k][1] for k in diversity_data], width, label='Предыдущие триплеты')
    
    plt.xlabel('Метрика')
    plt.ylabel('Количество')
    plt.title('Сравнение разнообразия')
    plt.xticks(x, diversity_data.keys())
    plt.legend()
    
    # Сравнение качества
    plt.subplot(1, 2, 2)
    quality_data = {
        'Позитивное сходство': [new_quality['avg_pos_similarity'], prev_quality['avg_pos_similarity']],
        'Негативное сходство': [new_quality['avg_neg_similarity'], prev_quality['avg_neg_similarity']],
        'Разница': [new_quality['avg_similarity_diff'], prev_quality['avg_similarity_diff']]
    }
    
    x = np.arange(len(quality_data))
    
    plt.bar(x - width/2, [quality_data[k][0] for k in quality_data], width, label='Новые триплеты')
    plt.bar(x + width/2, [quality_data[k][1] for k in quality_data], width, label='Предыдущие триплеты')
    
    plt.xlabel('Метрика')
    plt.ylabel('Значение')
    plt.title('Сравнение качества')
    plt.xticks(x, quality_data.keys())
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'triplet_comparison.png'))
    
    return comparison

# Функция для оценки контекстной релевантности триплетов
def evaluate_contextual_relevance(triplets, sample_size=200):
    # Если триплетов больше sample_size, выбираем случайную выборку
    if len(triplets) > sample_size:
        sampled_triplets = random.sample(triplets, sample_size)
    else:
        sampled_triplets = triplets
    
    # Группируем триплеты по доменам
    domain_triplets = {}
    for triplet in sampled_triplets:
        domain = triplet.get('domain')
        if domain not in domain_triplets:
            domain_triplets[domain] = []
        domain_triplets[domain].append(triplet)
    
    # Оцениваем контекстную релевантность для каждого домена
    domain_relevance = {}
    for domain, domain_triplets_list in domain_triplets.items():
        if not domain_triplets_list:
            continue
        
        # Извлекаем якоря и позитивные примеры
        anchors = [t['anchor'] for t in domain_triplets_list]
        positives = [t['positive'] for t in domain_triplets_list]
        
        # Создаем эмбеддинги
        anchor_embeddings = create_bert_embeddings(anchors)
        positive_embeddings = create_bert_embeddings(positives)
        
        # Вычисляем косинусное сходство между якорями и позитивными примерами
        similarities = []
        for i in range(len(anchor_embeddings)):
            sim = cosine_similarity([anchor_embeddings[i]], [positive_embeddings[i]])[0][0]
            similarities.append(sim)
        
        # Вычисляем среднее сходство для домена
        avg_similarity = np.mean(similarities)
        domain_relevance[domain] = avg_similarity
    
    # Визуализируем результаты
    plt.figure(figsize=(12, 6))
    plt.bar(domain_relevance.keys(), domain_relevance.values())
    plt.title('Контекстная релевантность триплетов по доменам')
    plt.xlabel('Домен')
    plt.ylabel('Среднее косинусное сходство')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'visualizations', 'contextual_relevance_by_domain.png'))
    
    return domain_relevance

# Функция для создания итогового отчета
def create_evaluation_report(evaluation_results):
    report = """# Отчет об оценке улучшенного механизма генерации триплетов

## 1. Обзор

Данный отчет представляет результаты оценки улучшенного механизма генерации триплетов для ИИ-модели бизнес-консультанта. Оценка проводилась по нескольким ключевым метрикам, включая разнообразие триплетов, качество триплетов, эффективность по типам и контекстную релевантность.

## 2. Разнообразие триплетов

### 2.1 Распределение по типам

"""
    
    # Добавляем информацию о распределении по типам
    type_distribution = evaluation_results['diversity']['type_distribution']
    report += "| Тип триплета | Количество |\n"
    report += "| --- | --- |\n"
    for triplet_type, count in type_distribution.items():
        report += f"| {triplet_type} | {count} |\n"
    
    report += f"\nВсего типов триплетов: {evaluation_results['diversity']['type_diversity']}\n\n"
    
    report += "### 2.2 Распределение по доменам\n\n"
    
    # Добавляем информацию о распределении по доменам
    domain_distribution = evaluation_results['diversity']['domain_distribution']
    report += "| Домен | Количество |\n"
    report += "| --- | --- |\n"
    for domain, count in domain_distribution.items():
        report += f"| {domain} | {count} |\n"
    
    report += f"\nВсего доменов: {evaluation_results['diversity']['domain_diversity']}\n\n"
    
    report += "## 3. Качество триплетов\n\n"
    
    # Добавляем информацию о качестве триплетов
    quality = evaluation_results['quality']
    report += f"Среднее косинусное сходство между якорями и позитивными примерами: {quality['avg_pos_similarity']:.4f}\n\n"
    report += f"Среднее косинусное сходство между якорями и негативными примерами: {quality['avg_neg_similarity']:.4f}\n\n"
    report += f"Средняя разница в сходстве (позитивные - негативные): {quality['avg_similarity_diff']:.4f}\n\n"
    
    report += "## 4. Эффективность по типам триплетов\n\n"
    
    # Добавляем информацию об эффективности по типам
    type_quality = evaluation_results['type_quality']
    report += "| Тип триплета | Позитивное сходство | Негативное сходство | Разница |\n"
    report += "| --- | --- | --- | --- |\n"
    for triplet_type, metrics in type_quality.items():
        report += f"| {triplet_type} | {metrics['avg_pos_similarity']:.4f} | {metrics['avg_neg_similarity']:.4f} | {metrics['avg_similarity_diff']:.4f} |\n"
    
    report += "\n## 5. Контекстная релевантность\n\n"
    
    # Добавляем информацию о контекстной релевантности
    domain_relevance = evaluation_results['contextual_relevance']
    report += "| Домен | Контекстная релевантность |\n"
    report += "| --- | --- |\n"
    for domain, relevance in domain_relevance.items():
        report += f"| {domain} | {relevance:.4f} |\n"
    
    # Если есть сравнение с предыдущей версией
    if 'comparison' in evaluation_results:
        report += "\n## 6. Сравнение с предыдущей версией\n\n"
        
        comparison = evaluation_results['comparison']
        
        report += "### 6.1 Сравнение разнообразия\n\n"
        report += "| Метрика | Новые триплеты | Предыдущие триплеты | Изменение |\n"
        report += "| --- | --- | --- | --- |\n"
        
        diversity_comp = comparison['diversity_comparison']
        report += f"| Типы триплетов | {diversity_comp['new_type_diversity']} | {diversity_comp['prev_type_diversity']} | {diversity_comp['new_type_diversity'] - diversity_comp['prev_type_diversity']} |\n"
        report += f"| Домены | {diversity_comp['new_domain_diversity']} | {diversity_comp['prev_domain_diversity']} | {diversity_comp['new_domain_diversity'] - diversity_comp['prev_domain_diversity']} |\n"
        
        report += "\n### 6.2 Сравнение качества\n\n"
        report += "| Метрика | Новые триплеты | Предыдущие триплеты | Изменение |\n"
        report += "| --- | --- | --- | --- |\n"
        
        quality_comp = comparison['quality_comparison']
        report += f"| Позитивное сходство | {quality_comp['new_avg_pos_similarity']:.4f} | {quality_comp['prev_avg_pos_similarity']:.4f} | {quality_comp['new_avg_pos_similarity'] - quality_comp['prev_avg_pos_similarity']:.4f} |\n"
        report += f"| Негативное сходство | {quality_comp['new_avg_neg_similarity']:.4f} | {quality_comp['prev_avg_neg_similarity']:.4f} | {quality_comp['new_avg_neg_similarity'] - quality_comp['prev_avg_neg_similarity']:.4f} |\n"
        report += f"| Разница | {quality_comp['new_avg_similarity_diff']:.4f} | {quality_comp['prev_avg_similarity_diff']:.4f} | {quality_comp['new_avg_similarity_diff'] - quality_comp['prev_avg_similarity_diff']:.4f} |\n"
    
    report += "\n## 7. Заключение\n\n"
    
    # Формируем заключение на основе результатов
    avg_similarity_diff = quality['avg_similarity_diff']
    if avg_similarity_diff > 0.3:
        quality_conclusion = "высоким"
    elif avg_similarity_diff > 0.2:
        quality_conclusion = "хорошим"
    elif avg_similarity_diff > 0.1:
        quality_conclusion = "удовлетворительным"
    else:
        quality_conclusion = "низким"
    
    report += f"Улучшенный механизм генерации триплетов демонстрирует {quality_conclusion} качество триплетов с средней разницей в сходстве {avg_similarity_diff:.4f}. "
    
    # Если есть сравнение с предыдущей версией
    if 'comparison' in evaluation_results:
        quality_comp = comparison['quality_comparison']
        diff_change = quality_comp['new_avg_similarity_diff'] - quality_comp['prev_avg_similarity_diff']
        
        if diff_change > 0.1:
            comparison_conclusion = "значительное улучшение"
        elif diff_change > 0.05:
            comparison_conclusion = "заметное улучшение"
        elif diff_change > 0:
            comparison_conclusion = "небольшое улучшение"
        else:
            comparison_conclusion = "отсутствие улучшения"
        
        report += f"По сравнению с предыдущей версией наблюдается {comparison_conclusion} в качестве триплетов. "
    
    # Добавляем рекомендации
    report += "\n\n### Рекомендации:\n\n"
    
    # Анализируем результаты по типам триплетов
    low_quality_types = []
    for triplet_type, metrics in type_quality.items():
        if metrics['avg_similarity_diff'] < 0.1:
            low_quality_types.append(triplet_type)
    
    if low_quality_types:
        report += f"1. Улучшить качество триплетов следующих типов: {', '.join(low_quality_types)}.\n"
    
    # Анализируем результаты по доменам
    low_relevance_domains = []
    for domain, relevance in domain_relevance.items():
        if relevance < 0.5:
            low_relevance_domains.append(domain)
    
    if low_relevance_domains:
        report += f"2. Повысить контекстную релевантность триплетов для следующих доменов: {', '.join(low_relevance_domains)}.\n"
    
    report += "3. Продолжить расширение разнообразия триплетов, особенно для междоменных связей.\n"
    report += "4. Внедрить механизм обратной связи от модели для динамической корректировки генерации триплетов.\n"
    
    return report

# Основная функция
def main():
    import random
    
    print("Запуск оценки улучшенного механизма генерации триплетов...")
    
    # Загружаем новые триплеты
    new_triplets_file = os.path.join(TRIPLETS_DIR, 'enhanced_triplets.json')
    if not os.path.exists(new_triplets_file):
        print(f"Файл с триплетами {new_triplets_file} не найден")
        return
    
    new_triplets = load_triplets(new_triplets_file)
    print(f"Загружено {len(new_triplets)} триплетов")
    
    # Оцениваем разнообразие триплетов
    print("Оценка разнообразия триплетов...")
    diversity_results = evaluate_triplet_diversity(new_triplets)
    
    # Оцениваем качество триплетов (на выборке)
    print("Оценка качества триплетов...")
    sample_size = min(300, len(new_triplets))
    sampled_triplets = random.sample(new_triplets, sample_size)
    quality_results = evaluate_triplet_quality(sampled_triplets)
    
    # Оцениваем эффективность по типам триплетов
    print("Оценка эффективности по типам триплетов...")
    type_quality_results = evaluate_triplet_effectiveness_by_type(new_triplets)
    
    # Оцениваем контекстную релевантность
    print("Оценка контекстной релевантности...")
    contextual_relevance_results = evaluate_contextual_relevance(new_triplets)
    
    # Сравниваем с предыдущей версией триплетов, если она доступна
    previous_triplets_file = '/home/ubuntu/ai_business_consultant/training_pipeline/triplets/triplets.json'
    comparison_results = None
    if os.path.exists(previous_triplets_file):
        print("Сравнение с предыдущей версией триплетов...")
        comparison_results = compare_with_previous_triplets(new_triplets, previous_triplets_file)
    
    # Собираем все результаты
    evaluation_results = {
        'diversity': diversity_results,
        'quality': quality_results,
        'type_quality': type_quality_results,
        'contextual_relevance': contextual_relevance_results
    }
    
    if comparison_results:
        evaluation_results['comparison'] = comparison_results
    
    # Создаем отчет
    print("Создание отчета об оценке...")
    report = create_evaluation_report(evaluation_results)
    
    # Сохраняем отчет
    report_file = os.path.join(RESULTS_DIR, 'evaluation_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Сохраняем результаты в JSON
    results_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        # Преобразуем numpy массивы в списки для сериализации
        serializable_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                        serializable_results[key][k] = [arr.tolist() for arr in v]
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"Отчет сохранен в {report_file}")
    print(f"Результаты сохранены в {results_file}")
    print("Оценка завершена.")

if __name__ == "__main__":
    main()
