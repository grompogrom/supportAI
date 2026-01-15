# Промпт для имплементации: LLM Client

## Задача
Реализовать клиент для работы с Perplexity API (модель sonar-pro).

## Файл
`src/llm_client.py`

## Контекст
Этот модуль является центральным для взаимодействия с LLM. Он должен отправлять запросы к Perplexity API, управлять историей сообщений и обрабатывать ответы.

## Требования к имплементации

### Класс LLMClient

#### `__init__(self, api_key: str, system_prompt: str)`
```
Реализуй инициализацию клиента:
1. Сохрани api_key в self._api_key
2. Сохрани system_prompt в self._system_prompt
3. Инициализируй пустой список self._messages для истории
4. Добавь системный промпт как первое сообщение: {"role": "system", "content": system_prompt}
5. Создай заголовки для HTTP запросов:
   - "Authorization": f"Bearer {api_key}"
   - "Content-Type": "application/json"
```

#### `send_message(self, message: str) -> str`
```
Реализуй отправку сообщения в LLM:
1. Добавь сообщение пользователя в историю: {"role": "user", "content": message}
2. Сформируй payload через _build_request_payload()
3. Отправь POST запрос на https://api.perplexity.ai/chat/completions
4. Проверь статус ответа, при ошибке вызови _handle_api_error()
5. Распарси ответ через _parse_response()
6. Добавь ответ ассистента в историю: {"role": "assistant", "content": response}
7. Верни текст ответа

Используй библиотеку requests для HTTP запросов.
```

#### `send_tool_result(self, tool_name: str, result: Any) -> str`
```
Реализуй отправку результата инструмента:
1. Сформируй сообщение с результатом в формате:
   "Результат выполнения инструмента {tool_name}:\n{json.dumps(result, ensure_ascii=False)}"
2. Добавь как сообщение с ролью "user" в историю
3. Отправь запрос к API (аналогично send_message)
4. Верни новый ответ модели
```

#### `_build_request_payload(self) -> Dict[str, Any]`
```
Сформируй payload для API:
{
    "model": "sonar-pro",
    "messages": self._messages,
    "temperature": 0.7,
    "max_tokens": 2048
}
```

#### `_parse_response(self, response_json: Dict) -> str`
```
Извлеки текст ответа:
1. Проверь наличие ключа "choices" в ответе
2. Извлеки response_json["choices"][0]["message"]["content"]
3. При отсутствии нужных ключей подними ValueError
```

#### `_handle_api_error(self, status_code: int, response_text: str)`
```
Обработай ошибки API:
- 401: raise AuthenticationError("Неверный API ключ")
- 429: raise RateLimitError("Превышен лимит запросов")
- Остальные: raise APIError(f"Ошибка API: {status_code} - {response_text}")
```

#### `get_messages_history(self) -> List[Dict]`
```
Верни копию списка self._messages (не саму ссылку!)
```

#### `clear_history(self)`
```
Очисти историю, но сохрани системный промпт:
1. self._messages = [{"role": "system", "content": self._system_prompt}]
```

#### `set_system_prompt(self, prompt: str)`
```
Обнови системный промпт:
1. self._system_prompt = prompt
2. Обнови первый элемент в self._messages
```

## Пример использования
```python
client = LLMClient(api_key="pplx-xxx", system_prompt="Ты ассистент")
response = client.send_message("Привет!")
print(response)
```

## Зависимости
- requests
- json (стандартная библиотека)
- typing (стандартная библиотека)

## Тестирование
После реализации проверь:
1. Успешную отправку сообщения (нужен реальный API ключ)
2. Корректное сохранение истории
3. Обработку ошибок (неверный ключ, сетевые ошибки)
