# Архитектура LLM клиентов

## Обзор

Система поддержки LLM построена на основе паттерна Strategy с использованием абстрактного базового класса. Это позволяет легко переключаться между различными провайдерами LLM без изменения остального кода.

## Диаграмма классов

```
┌─────────────────────────┐
│   BaseLLMClient (ABC)   │
├─────────────────────────┤
│ + send_message()        │
│ + send_tool_result()    │
│ + get_messages_history()│
│ + clear_history()       │
│ + set_system_prompt()   │
└───────────┬─────────────┘
            │
            │ наследуют
            │
    ┌───────┴────────┐
    │                │
┌───▼────────────┐  ┌▼──────────────────┐
│ Perplexity     │  │ LocalLLMClient    │
│ Client         │  │                   │
├────────────────┤  ├───────────────────┤
│ - api_key      │  │ - host            │
│ - headers      │  │ - port            │
│                │  │ - model_name      │
│ API:           │  │ - temperature     │
│ sonar-pro      │  │                   │
│                │  │ API:              │
│                │  │ Ollama /api/chat  │
└────────────────┘  └───────────────────┘
```

## Компоненты

### 1. BaseLLMClient (Абстрактный базовый класс)

**Назначение:** Определяет общий интерфейс для всех LLM клиентов.

**Основные методы:**

```python
class BaseLLMClient(ABC):
    def __init__(self, system_prompt: str) -> None
    
    @abstractmethod
    def send_message(self, message: str) -> str
    
    @abstractmethod
    def send_tool_result(self, tool_name: str, result: Any) -> str
    
    def get_messages_history(self) -> List[Dict[str, str]]
    def clear_history(self) -> None
    def set_system_prompt(self, prompt: str) -> None
```

**Общая функциональность:**
- Управление историей сообщений (`_messages`)
- Хранение системного промпта (`_system_prompt`)
- Базовые операции с историей

### 2. PerplexityClient

**Назначение:** Клиент для работы с Perplexity API.

**Особенности:**
- Использует модель `sonar-pro`
- Требует API ключ
- Быстрая генерация (~50-100 токенов/сек)
- Платный сервис

**Конфигурация:**
```python
client = PerplexityClient(
    api_key="pplx-...",
    system_prompt="Ты полезный ассистент"
)
```

**Внутренняя реализация:**
```python
API_BASE_URL = "https://api.perplexity.ai"
MODEL_NAME = "sonar-pro"

payload = {
    "model": MODEL_NAME,
    "messages": self._messages,
    "temperature": 0.7,
    "max_tokens": 2048,
    "disable_search": True
}
```

### 3. LocalLLMClient

**Назначение:** Клиент для работы с локальными моделями через Ollama.

**Особенности:**
- Поддержка различных моделей (qwen3:8b, llama3, mistral, phi3)
- Бесплатный
- Работает оффлайн
- Медленнее (~10-30 токенов/сек)
- Требует локальную установку Ollama

**Конфигурация:**
```python
client = LocalLLMClient(
    host="localhost",
    port=11434,
    model_name="qwen3:8b",
    system_prompt="Ты полезный ассистент",
    temperature=0.7
)
```

**Внутренняя реализация:**
```python
base_url = f"http://{host}:{port}/api/chat"

payload = {
    "model": model_name,
    "messages": self._messages,
    "stream": False,
    "options": {
        "temperature": temperature
    }
}
```

**Дополнительные методы:**
- `check_model_availability()` - проверка доступности модели

## Поток данных

### Отправка сообщения

```
User Input
    │
    ▼
main.py: send_to_llm()
    │
    ▼
BaseLLMClient: send_message()
    │
    ├─► PerplexityClient          ├─► LocalLLMClient
    │   │                          │   │
    │   ▼                          │   ▼
    │   POST /chat/completions     │   POST /api/chat
    │   │                          │   │
    │   ▼                          │   ▼
    │   Perplexity API             │   Ollama Server
    │   │                          │   │
    │   ▼                          │   ▼
    │   Response JSON              │   Response JSON
    │   │                          │   │
    └───┴──────────────────────────┴───┘
            │
            ▼
    _parse_response()
            │
            ▼
    Update _messages history
            │
            ▼
    Return response text
```

### Обработка Tool Results

```
Tool Execution Result
    │
    ▼
main.py: send_tool_result()
    │
    ▼
Format result as message
    │
    ▼
BaseLLMClient: send_tool_result()
    │
    ▼
[Same flow as send_message]
```

## Управление конфигурацией

### Выбор провайдера

**config/api_keys.yaml:**
```yaml
llm_provider: "perplexity"  # или "local"
```

**main.py:**
```python
def _create_llm_client(self, system_prompt: str) -> BaseLLMClient:
    provider = self._api_config.get('llm_provider', 'perplexity').lower()
    
    if provider == 'local':
        return LocalLLMClient(...)
    else:
        return PerplexityClient(...)
```

### Настройки локальной модели

**config/local_llm_config.yaml:**
```yaml
chat_model:
  host: "localhost"
  port: 11434
  model_name: "qwen3:8b"
  temperature: 0.7
```

## Обработка ошибок

### Иерархия исключений

```
Exception
    │
    └─► LLMError (базовая ошибка)
            │
            ├─► AuthenticationError (неверный API ключ)
            ├─► RateLimitError (превышен лимит)
            ├─► APIError (общая ошибка API)
            │
            └─► LocalLLMError (ошибки локальной LLM)
                    │
                    └─► LocalLLMConnectionError (ошибка подключения)
```

### Примеры обработки

**Perplexity:**
```python
try:
    response = client.send_message("Hello")
except AuthenticationError:
    print("Неверный API ключ")
except RateLimitError:
    print("Превышен лимит запросов")
except APIError as e:
    print(f"Ошибка API: {e}")
```

**LocalLLM:**
```python
try:
    response = client.send_message("Hello")
except LocalLLMConnectionError:
    print("Не удалось подключиться к Ollama")
    print("Убедитесь, что Ollama запущен")
except LocalLLMError as e:
    print(f"Ошибка локальной LLM: {e}")
```

## История сообщений

### Формат

```python
_messages = [
    {"role": "system", "content": "Системный промпт"},
    {"role": "user", "content": "Привет"},
    {"role": "assistant", "content": "Здравствуйте!"},
    {"role": "user", "content": "Как дела?"},
    {"role": "assistant", "content": "Отлично!"}
]
```

### Операции

**Получение истории:**
```python
history = client.get_messages_history()  # Возвращает копию
```

**Очистка истории:**
```python
client.clear_history()  # Сохраняет системный промпт
```

**Изменение промпта:**
```python
client.set_system_prompt("Новый промпт")
```

## Расширение системы

### Добавление нового провайдера

1. Создайте класс, наследующий `BaseLLMClient`:

```python
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, system_prompt: str):
        super().__init__(system_prompt)
        self._api_key = api_key
    
    def send_message(self, message: str) -> str:
        # Реализация для OpenAI API
        pass
    
    def send_tool_result(self, tool_name: str, result: Any) -> str:
        # Реализация
        pass
```

2. Добавьте в `_create_llm_client()`:

```python
elif provider == 'openai':
    return OpenAIClient(...)
```

3. Обновите конфигурацию:

```yaml
llm_provider: "openai"
```

## Тестирование

### Юнит-тесты

```python
def test_send_message():
    client = create_test_client("Системный промпт")
    response = client.send_message("Тест")
    assert len(response) > 0
    assert len(client.get_messages_history()) == 3
```

### Интеграционные тесты

```python
def test_local_llm_availability():
    client = LocalLLMClient(...)
    assert client.check_model_availability() == True
```

## Производительность

### Метрики

| Операция | Perplexity | LocalLLM |
|----------|------------|----------|
| Инициализация | ~0.1 сек | ~0.1 сек |
| Первый запрос | ~1-2 сек | ~5-10 сек |
| Последующие | ~1-2 сек | ~2-5 сек |
| Генерация (токены/сек) | 50-100 | 10-30 |

### Оптимизация

**LocalLLM:**
- Держите Ollama запущенным постоянно
- Используйте GPU если доступен
- Выбирайте модель по размеру задачи (phi3 для простых, qwen3:8b для сложных)

**Perplexity:**
- Используйте батчинг запросов
- Кэшируйте частые ответы
- Мониторьте rate limits

## Best Practices

1. **Выбор провайдера:**
   - Production: Perplexity (быстро, надежно)
   - Development: LocalLLM (бесплатно)
   - Приватные данные: только LocalLLM

2. **Управление историей:**
   - Очищайте историю при смене контекста
   - Ограничивайте длину истории (макс 10-20 сообщений)
   - Используйте `get_messages_history()` для отладки

3. **Обработка ошибок:**
   - Всегда оборачивайте вызовы в try-except
   - Логируйте ошибки для мониторинга
   - Имейте fallback стратегию

4. **Тестирование:**
   - Тестируйте с обоими провайдерами
   - Используйте моки для юнит-тестов
   - Проверяйте доступность перед интеграционными тестами

## Troubleshooting

### Perplexity не отвечает
- Проверьте API ключ
- Проверьте баланс аккаунта
- Проверьте rate limits

### LocalLLM не отвечает
- Проверьте, что Ollama запущен: `ollama list`
- Проверьте, что модель загружена: `ollama run qwen3:8b`
- Проверьте порт: `lsof -i :11434`

### Медленная генерация
- LocalLLM: используйте более легкую модель
- Perplexity: проверьте интернет соединение
- Оба: уменьшите max_tokens
