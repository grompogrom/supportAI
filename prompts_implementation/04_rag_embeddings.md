# Промпт для имплементации: RAG Embeddings

## Задача
Реализовать генератор эмбедингов через локальную LLM (Ollama или совместимый API).

## Файл
`src/rag/embeddings.py`

## Контекст
Этот модуль отвечает за преобразование текста в векторные представления (эмбединги) через локальную LLM. Эмбединги используются для семантического поиска.

## Требования к имплементации

### Класс EmbeddingGenerator

#### `__init__(self, config: EmbeddingConfig)`
```
Реализуй инициализацию:
1. Сохрани конфигурацию: self._config = config
2. Сформируй базовый URL:
   self._base_url = f"http://{config.host}:{config.port}{config.endpoint}"
3. Инициализируй размерность как None:
   self._embedding_dim = None
```

#### `generate(self, text: str) -> List[float]`
```
Реализуй генерацию эмбединга для одного текста:
1. Вызови _send_request(text)
2. Распарси ответ через _parse_embedding()
3. Верни список float

При ошибках используй _retry_with_backoff для повторных попыток.
```

#### `generate_batch(self, texts: List[str]) -> List[List[float]]`
```
Реализуй батчевую генерацию:
1. Создай список результатов: embeddings = []
2. Для каждого текста с индексом:
   for i, text in enumerate(texts):
       print(f"Генерация эмбединга {i+1}/{len(texts)}...")
       embedding = self.generate(text)
       embeddings.append(embedding)
3. Верни список эмбедингов

Примечание: Ollama не поддерживает батчевые запросы напрямую,
поэтому обрабатываем по одному.
```

#### `check_model_availability(self) -> bool`
```
Проверь доступность модели:
1. Попробуй сгенерировать эмбединг для тестовой строки "test"
2. Если успешно - верни True
3. При любой ошибке - верни False

try:
    self.generate("test")
    return True
except Exception:
    return False
```

#### `get_embedding_dimension(self) -> int`
```
Получи размерность эмбединга:
1. Если self._embedding_dim уже известна - верни её
2. Иначе сгенерируй тестовый эмбединг
3. Сохрани и верни длину вектора:
   self._embedding_dim = len(self.generate("test"))
   return self._embedding_dim
```

#### `_send_request(self, text: str) -> dict`
```
Отправь запрос к API локальной LLM:

Для Ollama API:
1. Сформируй payload:
   payload = {
       "model": self._config.model_name,
       "prompt": text
   }
2. Отправь POST запрос:
   response = requests.post(
       self._base_url,
       json=payload,
       timeout=self._config.timeout
   )
3. Проверь статус:
   if response.status_code != 200:
       raise EmbeddingConnectionError(f"Ошибка API: {response.status_code}")
4. Верни response.json()

При requests.exceptions.ConnectionError:
   raise EmbeddingConnectionError("Не удалось подключиться к LLM")
При requests.exceptions.Timeout:
   raise EmbeddingConnectionError("Таймаут подключения к LLM")
```

#### `_parse_embedding(self, response: dict) -> List[float]`
```
Извлеки эмбединг из ответа:

Для Ollama API формат ответа:
{
    "embedding": [0.123, -0.456, ...]
}

1. Проверь наличие ключа "embedding":
   if "embedding" not in response:
       raise EmbeddingParseError("Отсутствует поле 'embedding' в ответе")
2. Извлеки и верни:
   return response["embedding"]
```

#### `_retry_with_backoff(self, func: callable, *args, **kwargs)`
```
Реализуй повторные попытки с экспоненциальным backoff:

import time

max_attempts = self._config.retry_attempts
for attempt in range(max_attempts):
    try:
        return func(*args, **kwargs)
    except EmbeddingConnectionError as e:
        if attempt == max_attempts - 1:
            raise
        wait_time = 2 ** attempt  # 1, 2, 4 секунды
        print(f"Попытка {attempt + 1} не удалась. Ожидание {wait_time}с...")
        time.sleep(wait_time)
```

## Формат API Ollama для эмбедингов

Endpoint: `POST http://localhost:11434/api/embeddings`

Запрос:
```json
{
    "model": "nomic-embed-text",
    "prompt": "Текст для преобразования в эмбединг"
}
```

Ответ:
```json
{
    "embedding": [0.123, -0.456, 0.789, ...]
}
```

## Альтернативные модели для Ollama

- `nomic-embed-text` - рекомендуется, хороший баланс качества и скорости
- `mxbai-embed-large` - более точная, но медленнее
- `all-minilm` - быстрая, но менее точная

## Установка Ollama и модели

```bash
# Установка Ollama (macOS)
brew install ollama

# Запуск сервера
ollama serve

# Загрузка модели (в другом терминале)
ollama pull nomic-embed-text
```

## Зависимости
- requests
- time
- typing

## Тестирование
После реализации проверь:
1. Успешную генерацию эмбединга (нужен запущенный Ollama)
2. Корректную обработку ошибок подключения
3. Работу retry механизма
4. Батчевую обработку

Тест:
```python
config = EmbeddingConfig(
    host="localhost",
    port=11434,
    model_name="nomic-embed-text",
    endpoint="/api/embeddings"
)
generator = EmbeddingGenerator(config)
embedding = generator.generate("Привет мир")
print(f"Размерность: {len(embedding)}")
print(f"Первые 5 значений: {embedding[:5]}")
```
