// Основной JavaScript файл для пользовательского интерфейса ИИ Бизнес-Консультанта

// Конфигурация API
const API_CONFIG = {
    baseUrl: '/api',
    endpoints: {
        financial: '/financial',
        operations: '/operations',
        management: '/management',
        it: '/it',
        hr: '/hr',
        investment: '/investment',
        consulting: '/consulting'
    }
};

// Класс для работы с API
class ConsultingAPI {
    constructor(config) {
        this.config = config;
    }

    // Метод для отправки запроса на консультацию
    async getConsultation(domain, query, parameters = {}) {
        const endpoint = this.config.endpoints[domain] || this.config.endpoints.consulting;
        const url = this.config.baseUrl + endpoint;

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    domain: domain,
                    parameters: parameters
                })
            });

            if (!response.ok) {
                throw new Error(`Ошибка сервера: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Ошибка при получении консультации:', error);
            throw error;
        }
    }
}

// Класс для управления пользовательским интерфейсом
class ConsultingUI {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.initializeEventListeners();
    }

    // Инициализация обработчиков событий
    initializeEventListeners() {
        // Обработка выбора области консалтинга через карточки
        const domainCards = document.querySelectorAll('.domain-card');
        domainCards.forEach(card => {
            card.addEventListener('click', () => {
                const domain = card.getAttribute('data-domain');
                document.getElementById('domain-select').value = domain;
                document.getElementById('query-input').focus();
                
                // Прокрутка к форме запроса
                document.getElementById('consultation-title').scrollIntoView({ behavior: 'smooth' });
            });
        });
        
        // Обработка нажатия на кнопку "О проекте"
        const aboutBtn = document.getElementById('about-btn');
        aboutBtn.addEventListener('click', () => {
            const aboutModal = new bootstrap.Modal(document.getElementById('aboutModal'));
            aboutModal.show();
        });
        
        // Обработка отправки запроса
        const submitBtn = document.getElementById('submit-btn');
        submitBtn.addEventListener('click', () => this.sendConsultingRequest());
        
        // Обработка нажатия Enter в поле запроса
        const queryInput = document.getElementById('query-input');
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendConsultingRequest();
            }
        });
    }

    // Метод для отправки запроса на консультацию
    async sendConsultingRequest() {
        const query = document.getElementById('query-input').value.trim();
        const domain = document.getElementById('domain-select').value;
        
        if (!query) {
            this.showError('Пожалуйста, введите ваш запрос');
            return;
        }
        
        // Показываем индикатор загрузки
        this.showLoading(true);
        
        try {
            // Отправляем запрос на сервер
            const data = await this.apiClient.getConsultation(domain, query);
            
            // Отображаем результат
            this.displayResponse(data);
        } catch (error) {
            // Отображаем ошибку
            this.showError('Произошла ошибка при получении консультации: ' + error.message);
        } finally {
            // Скрываем индикатор загрузки
            this.showLoading(false);
        }
    }

    // Метод для отображения индикатора загрузки
    showLoading(isLoading) {
        document.getElementById('loading').style.display = isLoading ? 'block' : 'none';
        document.getElementById('response-container').style.display = isLoading ? 'none' : 'block';
    }

    // Метод для отображения ошибки
    showError(message) {
        alert(message);
    }

    // Метод для отображения результата
    displayResponse(data) {
        // Заполняем блок с ответом
        document.getElementById('response-text').textContent = data.result.analysis;
        
        // Заполняем список рекомендаций
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(recommendation => {
            const li = document.createElement('li');
            li.textContent = recommendation;
            recommendationsList.appendChild(li);
        });
        
        // Заполняем список источников
        const sourcesList = document.getElementById('sources-list');
        sourcesList.innerHTML = '';
        data.data_sources.forEach(source => {
            const li = document.createElement('li');
            if (source.startsWith('http')) {
                const a = document.createElement('a');
                a.href = source;
                a.textContent = source;
                a.target = '_blank';
                li.appendChild(a);
            } else {
                li.textContent = source;
            }
            sourcesList.appendChild(li);
        });
        
        // Заполняем дополнительную информацию
        document.getElementById('confidence-text').textContent = `Уверенность: ${(data.result.confidence * 100).toFixed(1)}%`;
        document.getElementById('timestamp-text').textContent = `Время запроса: ${new Date(data.timestamp).toLocaleString()}`;
        
        // Показываем блок с ответом
        document.getElementById('response-container').style.display = 'block';
        
        // Прокручиваем к блоку с ответом
        document.getElementById('response-container').scrollIntoView({ behavior: 'smooth' });
    }
}

// Инициализация приложения при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    // Создаем экземпляр API клиента
    const apiClient = new ConsultingAPI(API_CONFIG);
    
    // Создаем экземпляр UI
    const ui = new ConsultingUI(apiClient);
    
    console.log('ИИ Бизнес-Консультант инициализирован');
});
