import os
import pandas as pd
import numpy as np
import json
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertModel, BertTokenizer

# Пути к директориям с данными
DATA_ROOT = '/home/ubuntu/ai_business_consultant/triplet_improvement/datasets'
OUTPUT_DIR = '/home/ubuntu/ai_business_consultant/triplet_improvement/triplets'

# Создаем директорию для выходных данных, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка предобученной модели BERT для создания контекстуализированных эмбеддингов
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Устанавливаем модель в режим оценки

# Функция для создания эмбеддингов с помощью BERT
def create_bert_embeddings(texts, max_length=128):
    embeddings = []
    
    for text in tqdm(texts, desc="Creating BERT embeddings"):
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

# Функция для загрузки данных из всех источников
def load_all_datasets():
    datasets = {}
    
    # Загрузка данных IT Career Proficiency
    it_career_path = os.path.join(DATA_ROOT, 'it_career/it_career_data.csv')
    if os.path.exists(it_career_path):
        datasets['it_career'] = pd.read_csv(it_career_path)
    
    # Загрузка данных World Bank
    wb_paths = {
        'it': os.path.join(DATA_ROOT, 'world_bank/world_bank_it_data.csv'),
        'hr': os.path.join(DATA_ROOT, 'world_bank/world_bank_hr_data.csv'),
        'investment': os.path.join(DATA_ROOT, 'world_bank/world_bank_investment_data.csv'),
        'financial': os.path.join(DATA_ROOT, 'world_bank/world_bank_financial_data.csv'),
        'operations': os.path.join(DATA_ROOT, 'world_bank/world_bank_operations_data.csv'),
        'management': os.path.join(DATA_ROOT, 'world_bank/world_bank_management_data.csv')
    }
    
    for domain, path in wb_paths.items():
        if os.path.exists(path):
            datasets[f'world_bank_{domain}'] = pd.read_csv(path)
    
    # Загрузка данных IMF
    imf_paths = {
        'financial': os.path.join(DATA_ROOT, 'imf/imf_financial_data.csv'),
        'investment': os.path.join(DATA_ROOT, 'imf/imf_investment_data.csv'),
        'management': os.path.join(DATA_ROOT, 'imf/imf_management_data.csv')
    }
    
    for domain, path in imf_paths.items():
        if os.path.exists(path):
            datasets[f'imf_{domain}'] = pd.read_csv(path)
    
    # Загрузка данных Aria Title & Skill Taxonomy
    aria_path = os.path.join(DATA_ROOT, 'aria_taxonomy/aria_taxonomy_data.csv')
    if os.path.exists(aria_path):
        datasets['aria_taxonomy'] = pd.read_csv(aria_path)
    
    # Загрузка данных Bank Marketing
    bank_path = os.path.join(DATA_ROOT, 'bank_marketing/bank_marketing_data.csv')
    if os.path.exists(bank_path):
        datasets['bank_marketing'] = pd.read_csv(bank_path)
    
    # Загрузка данных Stock Market
    stock_path = os.path.join(DATA_ROOT, 'stock_market/stock_market_data.csv')
    if os.path.exists(stock_path):
        datasets['stock_market'] = pd.read_csv(stock_path)
    
    # Загрузка данных Quandl
    quandl_paths = {
        'financial': os.path.join(DATA_ROOT, 'quandl/quandl_financial_data.csv'),
        'investment': os.path.join(DATA_ROOT, 'quandl/quandl_investment_data.csv'),
        'operations': os.path.join(DATA_ROOT, 'quandl/quandl_operations_data.csv')
    }
    
    for domain, path in quandl_paths.items():
        if os.path.exists(path):
            datasets[f'quandl_{domain}'] = pd.read_csv(path)
    
    # Загрузка данных Hackett Group
    hackett_path = os.path.join(DATA_ROOT, 'hackett_group/hackett_group_data.csv')
    if os.path.exists(hackett_path):
        datasets['hackett_group'] = pd.read_csv(hackett_path)
    
    # Загрузка данных Qlik DataMarket
    qlik_path = os.path.join(DATA_ROOT, 'qlik_datamarket/qlik_datamarket_data.csv')
    if os.path.exists(qlik_path):
        datasets['qlik_datamarket'] = pd.read_csv(qlik_path)
    
    # Загрузка данных Atlas of Economic Complexity
    atlas_path = os.path.join(DATA_ROOT, 'economic_complexity/economic_complexity_index.csv')
    if os.path.exists(atlas_path):
        datasets['economic_complexity'] = pd.read_csv(atlas_path)
    
    # Загрузка данных Versatile Production System
    vps_path = os.path.join(DATA_ROOT, 'versatile_production/versatile_production_data.csv')
    if os.path.exists(vps_path):
        datasets['versatile_production'] = pd.read_csv(vps_path)
    
    return datasets

# Функция для генерации навыково-ориентированных триплетов
def generate_skill_oriented_triplets(datasets, num_triplets=1000):
    triplets = []
    
    # Используем данные IT Career Proficiency и Aria Taxonomy
    it_career_data = datasets.get('it_career')
    aria_data = datasets.get('aria_taxonomy')
    
    if it_career_data is None or aria_data is None:
        print("Отсутствуют необходимые данные для генерации навыково-ориентированных триплетов")
        return triplets
    
    # Объединяем данные из обоих источников
    roles = []
    skills = []
    
    # Обработка данных IT Career
    for _, row in it_career_data.iterrows():
        if 'role' in row and 'skills' in row:
            role = row['role']
            role_skills = row['skills'].split(',') if isinstance(row['skills'], str) else []
            
            for skill in role_skills:
                roles.append(role)
                skills.append(skill.strip())
    
    # Обработка данных Aria Taxonomy
    for _, row in aria_data.iterrows():
        if 'title' in row and 'skills' in row:
            role = row['title']
            role_skills = row['skills'].split(',') if isinstance(row['skills'], str) else []
            
            for skill in role_skills:
                roles.append(role)
                skills.append(skill.strip())
    
    # Создаем DataFrame для удобства обработки
    skill_data = pd.DataFrame({'role': roles, 'skill': skills})
    
    # Группируем навыки по ролям
    role_skills = {}
    for role in skill_data['role'].unique():
        role_skills[role] = skill_data[skill_data['role'] == role]['skill'].tolist()
    
    # Генерируем триплеты
    all_skills = skill_data['skill'].unique().tolist()
    
    for _ in tqdm(range(num_triplets), desc="Generating skill-oriented triplets"):
        # Выбираем роль с достаточным количеством навыков
        valid_roles = [r for r, s in role_skills.items() if len(s) >= 2]
        if not valid_roles:
            continue
        
        anchor_role = random.choice(valid_roles)
        positive_skills = role_skills[anchor_role]
        
        # Выбираем позитивный пример (навык, релевантный для роли)
        positive = random.choice(positive_skills)
        
        # Выбираем негативный пример (навык, не связанный с ролью)
        negative_skills = [s for s in all_skills if s not in positive_skills]
        if not negative_skills:
            continue
        
        negative = random.choice(negative_skills)
        
        # Создаем триплет
        triplet = {
            'type': 'skill_oriented',
            'domain': 'it' if 'IT' in anchor_role or 'Developer' in anchor_role else 'hr',
            'anchor': anchor_role,
            'positive': positive,
            'negative': negative,
            'metadata': {
                'anchor_type': 'role',
                'positive_type': 'skill',
                'negative_type': 'skill'
            }
        }
        
        triplets.append(triplet)
    
    return triplets

# Функция для генерации временных триплетов
def generate_temporal_triplets(datasets, num_triplets=1000):
    triplets = []
    
    # Используем данные World Bank, IMF, Quandl и Stock Market
    temporal_datasets = {k: v for k, v in datasets.items() if any(x in k for x in ['world_bank', 'imf', 'quandl', 'stock_market'])}
    
    if not temporal_datasets:
        print("Отсутствуют необходимые данные для генерации временных триплетов")
        return triplets
    
    # Для каждого домена создаем временные триплеты
    for dataset_name, dataset in temporal_datasets.items():
        # Определяем домен на основе имени датасета
        if 'it' in dataset_name:
            domain = 'it'
        elif 'hr' in dataset_name:
            domain = 'hr'
        elif 'investment' in dataset_name:
            domain = 'investment'
        elif 'financial' in dataset_name:
            domain = 'financial'
        elif 'operations' in dataset_name:
            domain = 'operations'
        else:
            domain = 'management'
        
        # Проверяем наличие временных столбцов
        time_columns = [col for col in dataset.columns if any(year in col for year in [str(y) for y in range(1990, 2025)])]
        
        if not time_columns or len(time_columns) < 2:
            continue
        
        # Сортируем временные столбцы
        time_columns.sort()
        
        # Получаем индикаторы/метрики
        indicator_columns = [col for col in dataset.columns if col not in time_columns and 'Unnamed' not in col]
        
        if not indicator_columns:
            continue
        
        # Генерируем триплеты для каждого индикатора
        for indicator in indicator_columns:
            # Пропускаем строки с пропущенными значениями
            valid_rows = dataset.dropna(subset=[indicator] + time_columns)
            
            if valid_rows.empty:
                continue
            
            for _ in range(min(num_triplets // len(temporal_datasets) // len(indicator_columns), 100)):
                # Выбираем случайную строку
                row = valid_rows.sample(1).iloc[0]
                
                # Выбираем случайный временной период (t)
                t_index = random.randint(0, len(time_columns) - 2)
                t_column = time_columns[t_index]
                t_plus_1_column = time_columns[t_index + 1]
                
                # Якорь: показатель в момент времени t
                anchor = f"{indicator} ({row[indicator]}) в {t_column}: {row[t_column]}"
                
                # Позитивный пример: тот же показатель в момент времени t+1
                positive = f"{indicator} ({row[indicator]}) в {t_plus_1_column}: {row[t_plus_1_column]}"
                
                # Негативный пример: другой показатель или тот же показатель для другой сущности
                other_rows = valid_rows[valid_rows.index != row.name]
                if not other_rows.empty:
                    other_row = other_rows.sample(1).iloc[0]
                    negative = f"{indicator} ({other_row[indicator]}) в {t_plus_1_column}: {other_row[t_plus_1_column]}"
                else:
                    # Если нет других строк, используем другой временной период
                    other_t_index = (t_index + 2) % len(time_columns)
                    other_t_column = time_columns[other_t_index]
                    negative = f"{indicator} ({row[indicator]}) в {other_t_column}: {row[other_t_column]}"
                
                # Создаем триплет
                triplet = {
                    'type': 'temporal',
                    'domain': domain,
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative,
                    'metadata': {
                        'indicator': indicator,
                        'entity': row[indicator] if isinstance(row[indicator], str) else str(row[indicator]),
                        't_period': t_column,
                        't_plus_1_period': t_plus_1_column
                    }
                }
                
                triplets.append(triplet)
    
    return triplets

# Функция для генерации контекстно-зависимых триплетов
def generate_context_dependent_triplets(datasets, num_triplets=1000):
    triplets = []
    
    # Используем данные Bank Marketing, Hackett Group и Versatile Production System
    context_datasets = {k: v for k, v in datasets.items() if any(x in k for x in ['bank_marketing', 'hackett_group', 'versatile_production'])}
    
    if not context_datasets:
        print("Отсутствуют необходимые данные для генерации контекстно-зависимых триплетов")
        return triplets
    
    # Для каждого датасета создаем контекстно-зависимые триплеты
    for dataset_name, dataset in context_datasets.items():
        # Определяем домен на основе имени датасета
        if 'bank_marketing' in dataset_name:
            domain = 'financial'
            context_column = 'customer_segment' if 'customer_segment' in dataset.columns else 'campaign_type'
            situation_column = 'situation' if 'situation' in dataset.columns else 'campaign_description'
            solution_column = 'solution' if 'solution' in dataset.columns else 'campaign_strategy'
            outcome_column = 'outcome' if 'outcome' in dataset.columns else 'campaign_result'
        elif 'hackett_group' in dataset_name:
            domain = 'operations'
            context_column = 'industry' if 'industry' in dataset.columns else 'company_size'
            situation_column = 'business_challenge' if 'business_challenge' in dataset.columns else 'process_issue'
            solution_column = 'solution_approach' if 'solution_approach' in dataset.columns else 'best_practice'
            outcome_column = 'outcome' if 'outcome' in dataset.columns else 'performance_improvement'
        else:  # versatile_production
            domain = 'operations'
            context_column = 'production_type' if 'production_type' in dataset.columns else 'industry'
            situation_column = 'production_challenge' if 'production_challenge' in dataset.columns else 'process_issue'
            solution_column = 'optimization_approach' if 'optimization_approach' in dataset.columns else 'solution'
            outcome_column = 'outcome' if 'outcome' in dataset.columns else 'efficiency_gain'
        
        # Проверяем наличие необходимых столбцов
        required_columns = [context_column, situation_column, solution_column, outcome_column]
        if not all(col in dataset.columns for col in required_columns):
            continue
        
        # Группируем решения по контекстам и ситуациям
        context_situations = {}
        for _, row in dataset.iterrows():
            context = row[context_column]
            situation = row[situation_column]
            solution = row[solution_column]
            outcome = row[outcome_column]
            
            key = (context, situation)
            if key not in context_situations:
                context_situations[key] = []
            
            context_situations[key].append((solution, outcome))
        
        # Генерируем триплеты
        for _ in tqdm(range(min(num_triplets // len(context_datasets), 300)), desc=f"Generating context-dependent triplets for {dataset_name}"):
            # Выбираем случайный контекст и ситуацию
            if not context_situations:
                continue
            
            (context, situation) = random.choice(list(context_situations.keys()))
            solutions = context_situations[(context, situation)]
            
            if len(solutions) < 1:
                continue
            
            # Якорь: бизнес-ситуация в контексте
            anchor = f"Контекст: {context}. Ситуация: {situation}"
            
            # Выбираем решение с лучшим исходом как позитивный пример
            # Предполагаем, что outcome - это числовое значение или строка, которую можно сравнить
            try:
                best_solution = max(solutions, key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) and x[1].replace('.', '', 1).isdigit() else 0)
                positive = f"Решение: {best_solution[0]}. Результат: {best_solution[1]}"
            except:
                # Если не удается определить лучшее решение, выбираем случайное
                solution, outcome = random.choice(solutions)
                positive = f"Решение: {solution}. Результат: {outcome}"
            
            # Негативный пример: решение из другого контекста/ситуации
            other_contexts = [k for k in context_situations.keys() if k != (context, situation)]
            if not other_contexts:
                # Если нет других контекстов, используем худшее решение из текущего контекста
                if len(solutions) > 1:
                    try:
                        worst_solution = min(solutions, key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) and x[1].replace('.', '', 1).isdigit() else 0)
                        negative = f"Решение: {worst_solution[0]}. Результат: {worst_solution[1]}"
                    except:
                        # Если не удается определить худшее решение, выбираем случайное отличное от позитивного
                        other_solutions = [s for s in solutions if f"Решение: {s[0]}. Результат: {s[1]}" != positive]
                        if other_solutions:
                            solution, outcome = random.choice(other_solutions)
                            negative = f"Решение: {solution}. Результат: {outcome}"
                        else:
                            # Если нет других решений, модифицируем позитивное решение
                            solution, outcome = random.choice(solutions)
                            negative = f"Решение: Противоположное к {solution}. Результат: Неудовлетворительный"
                else:
                    # Если есть только одно решение, создаем синтетический негативный пример
                    negative = f"Решение: Неоптимальный подход. Результат: Неудовлетворительный"
            else:
                # Выбираем решение из другого контекста
                other_context, other_situation = random.choice(other_contexts)
                other_solutions = context_situations[(other_context, other_situation)]
                if other_solutions:
                    solution, outcome = random.choice(other_solutions)
                    negative = f"Решение: {solution}. Результат: {outcome} (из контекста {other_context})"
                else:
                    negative = f"Решение: Неприменимое в данном контексте. Результат: Неудовлетворительный"
            
            # Создаем триплет
            triplet = {
                'type': 'context_dependent',
                'domain': domain,
                'anchor': anchor,
                'positive': positive,
                'negative': negative,
                'metadata': {
                    'context': context,
                    'situation': situation
                }
            }
            
            triplets.append(triplet)
    
    return triplets

# Функция для генерации экономико-сложностных триплетов
def generate_economic_complexity_triplets(datasets, num_triplets=1000):
    triplets = []
    
    # Используем данные Atlas of Economic Complexity и Qlik DataMarket
    atlas_data = datasets.get('economic_complexity')
    qlik_data = datasets.get('qlik_datamarket')
    
    if atlas_data is None or qlik_data is None:
        print("Отсутствуют необходимые данные для генерации экономико-сложностных триплетов")
        return triplets
    
    # Проверяем наличие необходимых столбцов
    required_atlas_columns = ['country', 'eci', 'year']
    required_qlik_columns = ['country', 'indicator', 'value', 'year']
    
    if not all(col in atlas_data.columns for col in required_atlas_columns) or not all(col in qlik_data.columns for col in required_qlik_columns):
        print("Отсутствуют необходимые столбцы для генерации экономико-сложностных триплетов")
        return triplets
    
    # Группируем данные по странам и годам
    country_eci = {}
    for _, row in atlas_data.iterrows():
        country = row['country']
        year = row['year']
        eci = row['eci']
        
        if country not in country_eci:
            country_eci[country] = {}
        
        country_eci[country][year] = eci
    
    # Группируем стратегии/индикаторы по странам и годам
    country_strategies = {}
    for _, row in qlik_data.iterrows():
        country = row['country']
        year = row['year']
        indicator = row['indicator']
        value = row['value']
        
        if country not in country_strategies:
            country_strategies[country] = {}
        
        if year not in country_strategies[country]:
            country_strategies[country][year] = []
        
        country_strategies[country][year].append((indicator, value))
    
    # Генерируем триплеты
    for _ in tqdm(range(num_triplets), desc="Generating economic complexity triplets"):
        # Выбираем случайную страну, которая есть в обоих датасетах
        common_countries = set(country_eci.keys()) & set(country_strategies.keys())
        if not common_countries:
            continue
        
        country = random.choice(list(common_countries))
        
        # Выбираем год, для которого есть данные в обоих датасетах
        common_years = set(country_eci[country].keys()) & set(country_strategies[country].keys())
        if not common_years:
            continue
        
        year = random.choice(list(common_years))
        eci = country_eci[country][year]
        
        # Якорь: страна с определенным индексом экономической сложности
        anchor = f"Страна: {country}, Год: {year}, Индекс экономической сложности: {eci}"
        
        # Выбираем стратегии/индикаторы для этой страны и года
        strategies = country_strategies[country][year]
        if not strategies:
            continue
        
        # Позитивный пример: стратегия, соответствующая уровню экономической сложности
        strategy, value = random.choice(strategies)
        positive = f"Стратегия: {strategy}, Значение: {value}"
        
        # Негативный пример: стратегия из страны с противоположным уровнем сложности
        # Находим страны с противоположным ECI (если ECI высокий, ищем низкий и наоборот)
        opposite_countries = []
        for c, years in country_eci.items():
            if c != country and year in years:
                if (eci > 0 and years[year] < 0) or (eci < 0 and years[year] > 0):
                    opposite_countries.append(c)
        
        if opposite_countries and all(c in country_strategies for c in opposite_countries):
            # Выбираем случайную страну с противоположным ECI
            opposite_country = random.choice(opposite_countries)
            
            # Проверяем, есть ли данные для этого года
            if year in country_strategies[opposite_country] and country_strategies[opposite_country][year]:
                opposite_strategy, opposite_value = random.choice(country_strategies[opposite_country][year])
                negative = f"Стратегия: {opposite_strategy}, Значение: {opposite_value} (из страны {opposite_country} с противоположным ECI)"
            else:
                # Если нет данных для этого года, используем другую стратегию из той же страны
                other_strategies = [s for s in strategies if s != (strategy, value)]
                if other_strategies:
                    other_strategy, other_value = random.choice(other_strategies)
                    negative = f"Стратегия: {other_strategy}, Значение: {other_value} (неоптимальная для данного ECI)"
                else:
                    negative = f"Стратегия: Противоположная к {strategy}, Значение: Неоптимальное"
        else:
            # Если нет стран с противоположным ECI, используем другую стратегию из той же страны
            other_strategies = [s for s in strategies if s != (strategy, value)]
            if other_strategies:
                other_strategy, other_value = random.choice(other_strategies)
                negative = f"Стратегия: {other_strategy}, Значение: {other_value} (менее оптимальная для данного ECI)"
            else:
                negative = f"Стратегия: Противоположная к {strategy}, Значение: Неоптимальное"
        
        # Определяем домен на основе типа стратегии/индикатора
        if 'investment' in strategy.lower() or 'fdi' in strategy.lower() or 'capital' in strategy.lower():
            domain = 'investment'
        elif 'finance' in strategy.lower() or 'banking' in strategy.lower() or 'monetary' in strategy.lower():
            domain = 'financial'
        elif 'management' in strategy.lower() or 'governance' in strategy.lower() or 'policy' in strategy.lower():
            domain = 'management'
        else:
            domain = 'investment'  # По умолчанию
        
        # Создаем триплет
        triplet = {
            'type': 'economic_complexity',
            'domain': domain,
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'metadata': {
                'country': country,
                'year': year,
                'eci': eci
            }
        }
        
        triplets.append(triplet)
    
    return triplets

# Функция для генерации производственно-ориентированных триплетов
def generate_production_oriented_triplets(datasets, num_triplets=1000):
    triplets = []
    
    # Используем данные Versatile Production System и Hackett Group
    vps_data = datasets.get('versatile_production')
    hackett_data = datasets.get('hackett_group')
    
    if vps_data is None and hackett_data is None:
        print("Отсутствуют необходимые данные для генерации производственно-ориентированных триплетов")
        return triplets
    
    # Объединяем данные из обоих источников
    production_processes = []
    optimization_solutions = []
    outcomes = []
    
    # Обработка данных Versatile Production System
    if vps_data is not None:
        process_col = 'production_process' if 'production_process' in vps_data.columns else 'process'
        solution_col = 'optimization_solution' if 'optimization_solution' in vps_data.columns else 'solution'
        outcome_col = 'efficiency_gain' if 'efficiency_gain' in vps_data.columns else 'outcome'
        
        if all(col in vps_data.columns for col in [process_col, solution_col, outcome_col]):
            for _, row in vps_data.iterrows():
                production_processes.append(row[process_col])
                optimization_solutions.append(row[solution_col])
                outcomes.append(row[outcome_col])
    
    # Обработка данных Hackett Group
    if hackett_data is not None:
        process_col = 'business_process' if 'business_process' in hackett_data.columns else 'process'
        solution_col = 'best_practice' if 'best_practice' in hackett_data.columns else 'solution'
        outcome_col = 'performance_improvement' if 'performance_improvement' in hackett_data.columns else 'outcome'
        
        if all(col in hackett_data.columns for col in [process_col, solution_col, outcome_col]):
            for _, row in hackett_data.iterrows():
                production_processes.append(row[process_col])
                optimization_solutions.append(row[solution_col])
                outcomes.append(row[outcome_col])
    
    # Проверяем, есть ли данные
    if not production_processes:
        print("Недостаточно данных для генерации производственно-ориентированных триплетов")
        return triplets
    
    # Создаем DataFrame для удобства обработки
    production_data = pd.DataFrame({
        'process': production_processes,
        'solution': optimization_solutions,
        'outcome': outcomes
    })
    
    # Группируем решения по процессам
    process_solutions = {}
    for _, row in production_data.iterrows():
        process = row['process']
        solution = row['solution']
        outcome = row['outcome']
        
        if process not in process_solutions:
            process_solutions[process] = []
        
        process_solutions[process].append((solution, outcome))
    
    # Генерируем триплеты
    for _ in tqdm(range(num_triplets), desc="Generating production-oriented triplets"):
        # Выбираем процесс с несколькими решениями
        valid_processes = [p for p, s in process_solutions.items() if len(s) >= 2]
        if not valid_processes:
            continue
        
        process = random.choice(valid_processes)
        solutions = process_solutions[process]
        
        # Якорь: производственный процесс
        anchor = f"Производственный процесс: {process}"
        
        # Пытаемся определить лучшее решение на основе outcome
        try:
            # Предполагаем, что outcome - это числовое значение или строка с числом
            solutions_with_numeric_outcome = []
            for sol, out in solutions:
                try:
                    if isinstance(out, str) and '%' in out:
                        # Если outcome содержит процент, извлекаем число
                        numeric_out = float(out.replace('%', '').strip())
                    else:
                        numeric_out = float(out)
                    solutions_with_numeric_outcome.append((sol, out, numeric_out))
                except:
                    pass
            
            if solutions_with_numeric_outcome:
                # Выбираем решение с наибольшим числовым outcome
                best_solution = max(solutions_with_numeric_outcome, key=lambda x: x[2])
                positive = f"Оптимизационное решение: {best_solution[0]}. Результат: {best_solution[1]}"
                
                # Выбираем решение с наименьшим числовым outcome
                worst_solution = min(solutions_with_numeric_outcome, key=lambda x: x[2])
                negative = f"Неоптимальное решение: {worst_solution[0]}. Результат: {worst_solution[1]}"
            else:
                # Если не удалось определить числовые outcome, выбираем случайно
                solution1, outcome1 = random.choice(solutions)
                positive = f"Оптимизационное решение: {solution1}. Результат: {outcome1}"
                
                # Выбираем другое решение для негативного примера
                other_solutions = [s for s in solutions if s != (solution1, outcome1)]
                if other_solutions:
                    solution2, outcome2 = random.choice(other_solutions)
                    negative = f"Альтернативное решение: {solution2}. Результат: {outcome2}"
                else:
                    negative = f"Неоптимальное решение: Противоположное к {solution1}. Результат: Неудовлетворительный"
        except:
            # Если не удалось определить лучшее/худшее решение, выбираем случайно
            solution1, outcome1 = random.choice(solutions)
            positive = f"Оптимизационное решение: {solution1}. Результат: {outcome1}"
            
            # Выбираем другое решение для негативного примера
            other_solutions = [s for s in solutions if s != (solution1, outcome1)]
            if other_solutions:
                solution2, outcome2 = random.choice(other_solutions)
                negative = f"Альтернативное решение: {solution2}. Результат: {outcome2}"
            else:
                negative = f"Неоптимальное решение: Противоположное к {solution1}. Результат: Неудовлетворительный"
        
        # Создаем триплет
        triplet = {
            'type': 'production_oriented',
            'domain': 'operations',
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'metadata': {
                'process': process
            }
        }
        
        triplets.append(triplet)
    
    return triplets

# Функция для балансировки триплетов между доменами
def balance_triplets_across_domains(triplets, target_per_domain=500):
    domains = ['it', 'hr', 'investment', 'financial', 'operations', 'management']
    balanced_triplets = []
    
    # Группируем триплеты по доменам
    domain_triplets = {domain: [] for domain in domains}
    for triplet in triplets:
        domain = triplet.get('domain')
        if domain in domains:
            domain_triplets[domain].append(triplet)
    
    # Балансируем количество триплетов в каждом домене
    for domain in domains:
        domain_count = len(domain_triplets[domain])
        
        if domain_count == 0:
            print(f"Предупреждение: нет триплетов для домена {domain}")
            continue
        
        if domain_count <= target_per_domain:
            # Если триплетов меньше целевого количества, добавляем все
            balanced_triplets.extend(domain_triplets[domain])
        else:
            # Если триплетов больше целевого количества, выбираем случайную выборку
            balanced_triplets.extend(random.sample(domain_triplets[domain], target_per_domain))
    
    return balanced_triplets

# Функция для генерации всех типов триплетов
def generate_all_triplets(num_triplets_per_type=1000):
    print("Загрузка всех наборов данных...")
    datasets = load_all_datasets()
    
    all_triplets = []
    
    print("Генерация навыково-ориентированных триплетов...")
    skill_triplets = generate_skill_oriented_triplets(datasets, num_triplets_per_type)
    all_triplets.extend(skill_triplets)
    print(f"Сгенерировано {len(skill_triplets)} навыково-ориентированных триплетов")
    
    print("Генерация временных триплетов...")
    temporal_triplets = generate_temporal_triplets(datasets, num_triplets_per_type)
    all_triplets.extend(temporal_triplets)
    print(f"Сгенерировано {len(temporal_triplets)} временных триплетов")
    
    print("Генерация контекстно-зависимых триплетов...")
    context_triplets = generate_context_dependent_triplets(datasets, num_triplets_per_type)
    all_triplets.extend(context_triplets)
    print(f"Сгенерировано {len(context_triplets)} контекстно-зависимых триплетов")
    
    print("Генерация экономико-сложностных триплетов...")
    economic_triplets = generate_economic_complexity_triplets(datasets, num_triplets_per_type)
    all_triplets.extend(economic_triplets)
    print(f"Сгенерировано {len(economic_triplets)} экономико-сложностных триплетов")
    
    print("Генерация производственно-ориентированных триплетов...")
    production_triplets = generate_production_oriented_triplets(datasets, num_triplets_per_type)
    all_triplets.extend(production_triplets)
    print(f"Сгенерировано {len(production_triplets)} производственно-ориентированных триплетов")
    
    print("Балансировка триплетов между доменами...")
    balanced_triplets = balance_triplets_across_domains(all_triplets)
    print(f"Итого сбалансированных триплетов: {len(balanced_triplets)}")
    
    return balanced_triplets

# Функция для сохранения триплетов в файл
def save_triplets(triplets, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"Триплеты сохранены в {output_file}")

# Основная функция
def main():
    print("Запуск генерации улучшенных триплетов...")
    
    # Генерируем триплеты всех типов
    triplets = generate_all_triplets(num_triplets_per_type=1000)
    
    # Сохраняем триплеты в файл
    output_file = os.path.join(OUTPUT_DIR, 'enhanced_triplets.json')
    save_triplets(triplets, output_file)
    
    # Сохраняем статистику по типам триплетов
    triplet_types = {}
    triplet_domains = {}
    
    for triplet in triplets:
        triplet_type = triplet.get('type')
        if triplet_type not in triplet_types:
            triplet_types[triplet_type] = 0
        triplet_types[triplet_type] += 1
        
        domain = triplet.get('domain')
        if domain not in triplet_domains:
            triplet_domains[domain] = 0
        triplet_domains[domain] += 1
    
    print("\nСтатистика по типам триплетов:")
    for triplet_type, count in triplet_types.items():
        print(f"{triplet_type}: {count}")
    
    print("\nСтатистика по доменам:")
    for domain, count in triplet_domains.items():
        print(f"{domain}: {count}")
    
    # Сохраняем статистику в файл
    stats = {
        'total_triplets': len(triplets),
        'triplet_types': triplet_types,
        'triplet_domains': triplet_domains
    }
    
    stats_file = os.path.join(OUTPUT_DIR, 'triplet_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Статистика сохранена в {stats_file}")
    print("Генерация триплетов завершена.")

if __name__ == "__main__":
    main()
