# Промпт для имплементации: MCP Handler

## Задача
Реализовать обработчик MCP (Model Context Protocol) инструментов для вызова внешних сервисов и локальных функций.

## Файл
`src/mcp_handler.py`

## Контекст
Этот модуль парсит ответы LLM на наличие вызовов инструментов, выполняет их через MCP серверы или локально, и возвращает результаты.

## Требования к имплементации

### Класс MCPHandler

#### `__init__(self, servers_config: Dict[str, MCPServerConfig])`
```
Реализуй инициализацию:
1. Сохрани servers_config в self._servers
2. Создай словарь self._local_tools = {} для локальных инструментов
3. Создай маппинг инструментов на серверы self._tool_to_server = {}
   - По умолчанию все MCP инструменты привязаны к "ticket_service"
   - Можно расширить для других серверов

Примечание: пока маппинг можно захардкодить:
- "create_ticket" -> "ticket_service"
- "get_ticket_status" -> "ticket_service"
- "search_knowledge_base" -> None (локальный)
```

#### `call_tool(self, request: ToolCallRequest) -> ToolCallResult`
```
Реализуй вызов инструмента:
1. Проверь, есть ли инструмент в self._local_tools
   - Если да: вызови локальный обработчик
2. Иначе найди сервер через _get_server_for_tool()
3. Если сервер найден: вызови _send_mcp_request()
4. Оберни результат в ToolCallResult(success=True, result=...)
5. При ошибках верни ToolCallResult(success=False, error_message=str(e))

Используй try-except для обработки исключений.
```

#### `parse_tool_call(self, llm_response: str) -> Optional[ToolCallRequest]`
```
Реализуй парсинг вызова инструмента:
1. Используй регулярное выражение для поиска:
   pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
2. Найди первое совпадение (re.search с флагом re.DOTALL)
3. Если не найдено - верни None
4. Извлеки JSON из найденного блока
5. Распарси JSON через json.loads()
6. Создай и верни ToolCallRequest:
   - tool_name = data["tool"]
   - parameters = data["parameters"]
7. При ошибках парсинга верни None
```

#### `format_tool_result(self, tool_name: str, result: ToolCallResult) -> str`
```
Сформируй строку результата:

Если result.success:
<tool_result>
{
  "tool": "{tool_name}",
  "success": true,
  "result": {result.result}
}
</tool_result>

Иначе:
<tool_result>
{
  "tool": "{tool_name}",
  "success": false,
  "error": "{result.error_message}"
}
</tool_result>

Используй json.dumps с ensure_ascii=False и indent=2
```

#### `has_tool_call(self, llm_response: str) -> bool`
```
Проверь наличие тега <tool_call> в тексте:
return "<tool_call>" in llm_response and "</tool_call>" in llm_response
```

#### `register_local_tool(self, name: str, handler: callable, description: str, parameters: Dict)`
```
Зарегистрируй локальный инструмент:
self._local_tools[name] = {
    "handler": handler,
    "description": description,
    "parameters": parameters
}
```

#### `get_available_tools(self) -> List[Dict]`
```
Верни список всех доступных инструментов:
1. Собери инструменты из MCP серверов (можно захардкодить список)
2. Добавь локальные инструменты из self._local_tools
3. Верни объединенный список
```

#### `_send_mcp_request(self, server: MCPServerConfig, tool_name: str, params: Dict) -> Dict`
```
Отправь запрос на MCP сервер:
1. Сформируй URL: f"http://{server.host}:{server.port}{server.endpoint}"
2. Сформируй payload:
   {
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
           "name": tool_name,
           "arguments": params
       },
       "id": 1
   }
3. Отправь POST запрос с JSON
4. Проверь статус ответа
5. Верни response.json()["result"]

При ошибках подключения: raise MCPConnectionError
При ошибках инструмента: raise MCPToolError
```

#### `_get_server_for_tool(self, tool_name: str) -> Optional[MCPServerConfig]`
```
Найди сервер для инструмента:
1. Проверь self._tool_to_server.get(tool_name)
2. Если None - инструмент локальный, верни None
3. Иначе верни self._servers[server_name]
```

#### `check_servers_health(self) -> Dict[str, bool]`
```
Проверь доступность всех серверов:
1. Для каждого сервера попробуй отправить запрос на endpoint
2. Запиши результат: {server_name: True/False}
3. Используй timeout=5 секунд
4. Оберни в try-except для обработки ошибок подключения
```

## Формат MCP запроса/ответа

Запрос:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "create_ticket",
    "arguments": {
      "user_name": "Иван",
      "issue_summary": "Проблема с входом"
    }
  },
  "id": 1
}
```

Ответ:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "ticket_id": "TKT-12345",
    "status": "created"
  },
  "id": 1
}
```

## Зависимости
- requests
- json
- re (регулярные выражения)
- typing

## Тестирование
После реализации проверь:
1. Корректный парсинг tool_call из строки
2. Форматирование результата
3. Регистрацию локального инструмента
4. Обработку ошибок при недоступности сервера
