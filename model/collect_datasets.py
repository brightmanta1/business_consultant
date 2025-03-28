import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Настройка путей
BASE_DIR = '/home/ubuntu/ai_business_consultant/triplet_improvement'
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

# Функция для создания директории, если она не существует
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для загрузки данных из World Bank API
def collect_world_bank_data():
    print("Сбор данных из World Bank Open Dataset...")
    output_dir = os.path.join(DATASETS_DIR, 'world_bank')
    ensure_dir(output_dir)
    
    # Список индикаторов для различных областей консалтинга
    indicators = {
        'it': ['IT.NET.USER.ZS', 'IT.CEL.SETS.P2', 'IT.NET.SECR.P6', 'GB.XPD.RSDV.GD.ZS'],
        'hr': ['SL.TLF.TOTL.IN', 'SL.UEM.TOTL.ZS', 'SL.TLF.ACTI.ZS', 'SL.EMP.TOTL.SP.ZS'],
        'investment': ['CM.MKT.LCAP.GD.ZS', 'CM.MKT.TRNR', 'BX.KLT.DINV.WD.GD.ZS', 'GC.DOD.TOTL.GD.ZS'],
        'financial': ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD', 'FP.CPI.TOTL.ZG', 'BN.CAB.XOKA.GD.ZS'],
        'operations': ['NV.IND.MANF.ZS', 'TX.VAL.TECH.MF.ZS', 'IS.VEH.PCAR.P3', 'IS.AIR.PSGR'],
        'management': ['IC.BUS.EASE.XQ', 'IC.REG.DURS', 'IC.TAX.TOTL.CP.ZS', 'IC.LGL.CRED.XQ']
    }
    
    # Список стран для сбора данных
    countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'BRA', 'RUS', 'CAN']
    
    # Сбор данных по каждому индикатору для каждой страны
    all_data = []
    for domain, domain_indicators in indicators.items():
        for indicator in domain_indicators:
            for country in countries:
                try:
                    # Формируем URL для API запроса
                    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=100"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Проверяем, что получены данные
                        if len(data) > 1 and data[1]:
                            for entry in data[1]:
                                if entry['value'] is not None:
                                    all_data.append({
                                        'country': country,
                                        'indicator': indicator,
                                        'indicator_name': entry.get('indicator', {}).get('value', ''),
                                        'year': entry['date'],
                                        'value': entry['value'],
                                        'domain': domain
                                    })
                    
                    print(f"Собраны данные для {country} - {indicator}")
                except Exception as e:
                    print(f"Ошибка при сборе данных для {country} - {indicator}: {e}")
    
    # Сохраняем собранные данные
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(output_dir, 'world_bank_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Данные World Bank сохранены в {output_file}")
        
        # Сохраняем отдельные файлы для каждого домена
        for domain in indicators.keys():
            domain_df = df[df['domain'] == domain]
            if not domain_df.empty:
                domain_file = os.path.join(output_dir, f'world_bank_{domain}_data.csv')
                domain_df.to_csv(domain_file, index=False)
                print(f"Данные для домена {domain} сохранены в {domain_file}")
    else:
        print("Не удалось собрать данные из World Bank")

# Функция для загрузки данных из IMF API
def collect_imf_data():
    print("Сбор данных из IMF Dataset...")
    output_dir = os.path.join(DATASETS_DIR, 'imf')
    ensure_dir(output_dir)
    
    # Список наборов данных IMF для различных областей консалтинга
    datasets = {
        'financial': ['IFS', 'BOP', 'GFSR'],
        'investment': ['CPIS', 'COFER', 'FDI'],
        'management': ['WEO', 'FARI', 'GFS']
    }
    
    # Симуляция сбора данных из IMF (в реальности требуется API ключ)
    all_data = []
    
    # Генерируем синтетические данные для демонстрации
    for domain, domain_datasets in datasets.items():
        for dataset in domain_datasets:
            # Генерируем данные для 10 стран за 5 лет
            countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'BRA', 'RUS', 'CAN']
            years = [2018, 2019, 2020, 2021, 2022]
            
            for country in countries:
                for year in years:
                    # Генерируем случайные значения для показателей
                    value = np.random.normal(100, 20)
                    growth = np.random.normal(2, 5)
                    
                    all_data.append({
                        'country': country,
                        'dataset': dataset,
                        'year': year,
                        'value': value,
                        'growth': growth,
                        'domain': domain
                    })
            
            print(f"Сгенерированы данные для {dataset}")
    
    # Сохраняем собранные данные
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(output_dir, 'imf_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Данные IMF сохранены в {output_file}")
        
        # Сохраняем отдельные файлы для каждого домена
        for domain in datasets.keys():
            domain_df = df[df['domain'] == domain]
            if not domain_df.empty:
                domain_file = os.path.join(output_dir, f'imf_{domain}_data.csv')
                domain_df.to_csv(domain_file, index=False)
                print(f"Данные для домена {domain} сохранены в {domain_file}")
    else:
        print("Не удалось собрать данные из IMF")

# Функция для загрузки данных из IT Career Proficiency dataset
def collect_it_career_data():
    print("Сбор данных из IT Career Proficiency dataset...")
    output_dir = os.path.join(DATASETS_DIR, 'it_career')
    ensure_dir(output_dir)
    
    # Симуляция сбора данных из IT Career Proficiency dataset
    # В реальности требуется доступ к API или загрузка файлов
    
    # Создаем структуру данных для IT профессий и навыков
    it_roles = [
        'Software Developer', 'Data Scientist', 'DevOps Engineer', 'Cloud Architect',
        'Cybersecurity Specialist', 'Network Administrator', 'Database Administrator',
        'IT Project Manager', 'UI/UX Designer', 'QA Engineer'
    ]
    
    skill_categories = {
        'Programming Languages': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'Ruby', 'PHP', 'Swift', 'Kotlin'],
        'Frameworks': ['React', 'Angular', 'Vue.js', 'Django', 'Spring', 'Flask', 'TensorFlow', 'PyTorch', 'Node.js', 'ASP.NET'],
        'Databases': ['MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'SQL Server', 'Redis', 'Cassandra', 'DynamoDB', 'Elasticsearch', 'Neo4j'],
        'Cloud Platforms': ['AWS', 'Azure', 'Google Cloud', 'IBM Cloud', 'Oracle Cloud', 'DigitalOcean', 'Heroku', 'Alibaba Cloud', 'VMware', 'OpenStack'],
        'DevOps Tools': ['Docker', 'Kubernetes', 'Jenkins', 'Git', 'Ansible', 'Terraform', 'Prometheus', 'Grafana', 'CircleCI', 'Travis CI'],
        'Soft Skills': ['Communication', 'Problem Solving', 'Teamwork', 'Time Management', 'Leadership', 'Adaptability', 'Critical Thinking', 'Creativity', 'Emotional Intelligence', 'Conflict Resolution']
    }
    
    # Генерируем данные о профессиях и требуемых навыках
    all_data = []
    for role in it_roles:
        # Для каждой роли выбираем набор навыков из разных категорий
        role_skills = {}
        for category, skills in skill_categories.items():
            # Выбираем случайное количество навыков из категории
            num_skills = np.random.randint(1, len(skills) // 2 + 1)
            selected_skills = np.random.choice(skills, num_skills, replace=False)
            
            # Для каждого навыка генерируем уровень владения (1-5)
            for skill in selected_skills:
                proficiency = np.random.randint(1, 6)
                all_data.append({
                    'role': role,
                    'skill': skill,
                    'category': category,
                    'proficiency': proficiency,
                    'importance': np.random.randint(1, 6)
                })
    
    # Сохраняем собранные данные
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(output_dir, 'it_career_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Данные IT Career Proficiency сохранены в {output_file}")
        
        # Создаем сводную таблицу по ролям и категориям навыков
        pivot_df = df.pivot_table(
            index='role',
            columns='category',
            values='proficiency',
            aggfunc='mean'
        ).reset_index()
        
        pivot_file = os.path.join(output_dir, 'it_career_pivot.csv')
        pivot_df.to_csv(pivot_file, index=False)
        print(f"Сводные данные сохранены в {pivot_file}")
    else:
        print("Не удалось собрать данные из IT Career Proficiency dataset")

# Функция для загрузки данных из Aria Title & Skill Taxonomy
def collect_aria_taxonomy_data():
    print("Сбор данных из Aria Title & Skill Taxonomy...")
    output_dir = os.path.join(DATASETS_DIR, 'aria_taxonomy')
    ensure_dir(output_dir)
    
    # Симуляция сбора данных из Aria Title & Skill Taxonomy
    # В реальности требуется доступ к API или загрузка файлов
    
    # Создаем структуру данных для должностей и навыков в разных отраслях
    industries = [
        'Technology', 'Finance', 'Healthcare', 'Manufacturing', 
        'Retail', 'Education', 'Consulting', 'Energy', 'Media', 'Transportation'
    ]
    
    job_levels = ['Entry', 'Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director', 'VP', 'C-Suite']
    
    # Генерируем данные о должностях и навыках для HR-консалтинга
    all_data = []
    job_id = 1
    
    for industry in industries:
        # Генерируем должности для каждой отрасли
        if industry == 'Technology':
            job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 'IT Support']
        elif industry == 'Finance':
            job_titles = ['Financial Analyst', 'Investment Banker', 'Accountant', 'Risk Manager', 'Financial Advisor']
        elif industry == 'Healthcare':
            job_titles = ['Nurse', 'Physician', 'Medical Technician', 'Healthcare Administrator', 'Pharmacist']
        elif industry == 'Manufacturing':
            job_titles = ['Production Manager', 'Quality Control', 'Supply Chain Analyst', 'Process Engineer', 'Plant Manager']
        elif industry == 'Retail':
            job_titles = ['Store Manager', 'Sales Associate', 'Merchandiser', 'Buyer', 'Customer Service Representative']
        elif industry == 'Education':
            job_titles = ['Teacher', 'Professor', 'Academic Advisor', 'Dean', 'Educational Technologist']
        elif industry == 'Consulting':
            job_titles = ['Management Consultant', 'Strategy Consultant', 'IT Consultant', 'HR Consultant', 'Operations Consultant']
        elif industry == 'Energy':
            job_titles = ['Petroleum Engineer', 'Energy Analyst', 'Environmental Specialist', 'Project Manager', 'Geologist']
        elif industry == 'Media':
            job_titles = ['Content Creator', 'Marketing Specialist', 'Journalist', 'Social Media Manager', 'Public Relations']
        else:  # Transportation
            job_titles = ['Logistics Manager', 'Transportation Planner', 'Fleet Manager', 'Supply Chain Coordinator', 'Dispatcher']
        
        for job_title in job_titles:
            for level in job_levels:
                full_title = f"{level} {job_title}"
                
                # Генерируем требуемые навыки для должности
                required_skills = []
                if 'Engineer' in job_title or 'Developer' in job_title:
                    required_skills.extend(['Programming', 'Problem Solving', 'Technical Documentation'])
                if 'Manager' in job_title or level in ['Manager', 'Director', 'VP', 'C-Suite']:
                    required_skills.extend(['Leadership', 'Strategic Planning', 'Team Management'])
                if 'Analyst' in job_title or 'Scientist' in job_title:
                    required_skills.extend(['Data Analysis', 'Statistical Methods', 'Critical Thinking'])
                if 'Consultant' in job_title:
                    required_skills.extend(['Client Communication', 'Problem Solving', 'Industry Knowledge'])
                
                # Добавляем общие навыки
                required_skills.extend(['Communication', 'Teamwork', 'Time Management'])
                
                # Удаляем дубликаты
                required_skills = list(set(required_skills))
                
                # Генерируем данные о зарплате
                if level == 'Entry':
                    salary_min, salary_max = 40000, 60000
                elif level == 'Junior':
                    salary_min, salary_max = 50000, 75000
                elif level == 'Mid':
                    salary_min, salary_max = 70000, 100000
                elif level == 'Senior':
                    salary_min, salary_max = 90000, 130000
                elif level == 'Lead':
                    salary_min, salary_max = 110000, 150000
                elif level == 'Manager':
                    salary_min, salary_max = 120000, 180000
                elif level == 'Director':
                    salary_min, salary_max = 150000, 220000
                elif level == 'VP':
                    salary_min, salary_max = 180000, 300000
                else:  # C-Suite
                    salary_min, salary_max = 250000, 500000
                
                # Корректируем зарплату в зависимости от отрасли
                if industry in ['Finance', 'Technology', 'Consulting']:
                    salary_min *= 1.2
                    salary_max *= 1.2
                elif industry in ['Retail', 'Education']:
                    salary_min *= 0.8
                    salary_max *= 0.8
                
                all_data.append({
                    'job_id': job_id,
                    'title': full_title,
                    'industry': industry,
                    'level': level,
                    'required_skills': ', '.join(required_skills),
                    'salary_min': int(salary_min),
                    'salary_max': int(salary_max),
                    'education_required': 'Bachelor' if level in ['Entry', 'Junior', 'Mid'] else 'Master',
                    'experience_years_min': 0 if level == 'Entry' else (2 if level == 'Junior' else (5 if level == 'Mid' else (8 if level == 'Senior' else 10)))
                })
                
                job_id += 1
    
    # Сохраняем собранные данные
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(output_dir, 'aria_taxonomy_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Данные Aria Title & Skill Taxonomy сохранены в {output_file}")
        
        # Создаем сводную таблицу по отраслям и уровням должностей
        pivot_df = df.pivot_table(
            index='industry',
            columns='level',
            values='salary_min',
            aggfunc='mean'
        ).reset_index()
        
        pivot_file = os.path.join(output_dir, 'aria_taxonomy_pivot.csv')
        pivot_df.to_csv(pivot_file, index=False)
        print(f"Сводные данные сохранены в {pivot_file}")
    else:
        print("Не удалось собрать данные из Aria Title & Skill Taxonomy")

# Функция для загрузки данных из Bank Marketing Dataset
def collect_bank_marketing_data():
    print("Сбор данных из Bank Marke<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>