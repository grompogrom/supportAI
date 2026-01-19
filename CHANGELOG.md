# Changelog

## [Unreleased] - 2026-01-19

### Added

#### Поддержка локальных LLM моделей
- Добавлен `LocalLLMClient` для работы с локальными моделями через Ollama
- Поддержка модели qwen3:8b (и других: llama3, mistral, phi3)
- Возможность переключения между Perplexity API и локальной моделью через конфигурацию

#### Архитектурные улучшения
- Создан базовый абстрактный класс `BaseLLMClient`
- Рефакторинг существующего клиента в `PerplexityClient`
- Единый интерфейс для всех LLM провайдеров

#### Конфигурация
- Добавлен параметр `llm_provider` в `config/api_keys.yaml`
- Расширен `config/local_llm_config.yaml` с настройками для chat модели
- Поддержка настройки температуры генерации

#### Документация
- Создан `docs/local_llm_setup.md` - подробное руководство по настройке локальных моделей
- Создан `QUICKSTART.md` - быстрый старт для новых пользователей
- Обновлен `README.md` с информацией о поддержке локальных моделей
- Добавлены примеры использования в `examples/llm_client_usage.py`

#### Тесты
- Обновлены тесты для поддержки обоих типов клиентов
- Добавлена автоматическая проверка доступности локальной модели
- Тесты теперь работают с любым выбранным провайдером

### Changed

#### Breaking Changes
- `LLMClient` теперь является алиасом для `PerplexityClient` (для обратной совместимости)
- Рекомендуется использовать явные классы: `PerplexityClient` или `LocalLLMClient`

#### Улучшения в main.py
- Добавлен метод `_create_llm_client()` для динамического выбора провайдера
- Вывод информации о выбранном провайдере при запуске
- Проверка доступности локальной модели с предупреждениями

### Technical Details

#### Новые классы
```python
# Базовый класс
class BaseLLMClient(ABC):
    - send_message(message: str) -> str
    - send_tool_result(tool_name: str, result: Any) -> str
    - get_messages_history() -> List[Dict[str, str]]
    - clear_history() -> None
    - set_system_prompt(prompt: str) -> None

# Perplexity клиент
class PerplexityClient(BaseLLMClient):
    - API_BASE_URL = "https://api.perplexity.ai"
    - MODEL_NAME = "sonar-pro"

# Локальный клиент
class LocalLLMClient(BaseLLMClient):
    - Подключение к Ollama через http://localhost:11434/api/chat
    - Поддержка различных моделей
    - check_model_availability() -> bool
```

#### Новые исключения
- `LocalLLMError` - базовая ошибка локальной LLM
- `LocalLLMConnectionError` - ошибка подключения к Ollama

#### Конфигурационные файлы

**config/api_keys.yaml:**
```yaml
llm_provider: "perplexity"  # или "local"
```

**config/local_llm_config.yaml:**
```yaml
chat_model:
  host: "localhost"
  port: 11434
  model_name: "qwen3:8b"
  temperature: 0.7
```

### Performance

#### Сравнение производительности

| Метрика | Perplexity API | LocalLLM (qwen3:8b) |
|---------|----------------|---------------------|
| Скорость | ~50-100 токенов/сек | ~10-30 токенов/сек |
| Первый запрос | ~1-2 сек | ~5-10 сек |
| Последующие | ~1-2 сек | ~2-5 сек |
| Стоимость | Платно | Бесплатно |
| Требования RAM | - | 8 GB |
| Требования VRAM | - | 6 GB |

### Migration Guide

#### Для существующих пользователей

Если вы использовали `LLMClient` напрямую, код продолжит работать без изменений:

```python
# Старый код (работает)
from llm_client import LLMClient
client = LLMClient(api_key="...", system_prompt="...")

# Новый код (рекомендуется)
from llm_client import PerplexityClient
client = PerplexityClient(api_key="...", system_prompt="...")
```

#### Для перехода на локальную модель

1. Установите Ollama и загрузите модель
2. Измените `llm_provider: "local"` в `config/api_keys.yaml`
3. Перезапустите приложение

### Known Issues

- Локальные модели могут быть медленнее на первом запросе (загрузка в память)
- Требуется достаточно RAM/VRAM для работы модели
- Ollama должен быть запущен перед использованием локальной модели

### Future Plans

- [ ] Поддержка streaming ответов для локальных моделей
- [ ] Кэширование моделей в памяти
- [ ] Поддержка других провайдеров (OpenAI, Anthropic)
- [ ] Автоматическое переключение на fallback провайдер при недоступности основного
- [ ] Метрики и мониторинг производительности
