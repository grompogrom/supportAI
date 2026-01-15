# Промпт для имплементации: System Prompt

## Задача
Реализовать функции для работы с системным промптом.

## Файл
`src/prompts/system_prompt.py`

## Контекст
Системный промпт уже определен как константа `SYSTEM_PROMPT`. Нужно реализовать вспомогательные функции для динамической генерации и форматирования.

## Требования к имплементации

### Функция `get_system_prompt(tools_override: List[Dict] = None) -> str`
```
Реализуй получение системного промпта:

def get_system_prompt(tools_override: List[Dict[str, Any]] = None) -> str:
    """
    Получение системного промпта с возможностью кастомизации инструментов.
    """
    if tools_override is None:
        # Вернуть стандартный промпт без изменений
        return SYSTEM_PROMPT
    
    # Сгенерировать описание инструментов
    tools_description = format_tools_description(tools_override)
    
    # Заменить секцию инструментов в промпте
    # Найти и заменить блок между "## ДОСТУПНЫЕ ИНСТРУМЕНТЫ" и "## ФОРМАТ ВЫЗОВА"
    
    # Простой вариант - вернуть базовый промпт с добавлением инструментов
    base_prompt = '''
Ты - ассистент службы поддержки. Твоя задача - помогать пользователям решать их проблемы.

## ТВОИ ОБЯЗАННОСТИ:
1. Приветствовать пользователя и узнать его имя
2. Выяснить суть проблемы или вопроса
3. Создать тикет в системе поддержки
4. Найти релевантную информацию в базе знаний
5. Предоставить пользователю полезный ответ

## ДОСТУПНЫЕ ИНСТРУМЕНТЫ:

{tools}

## ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:
Когда тебе нужно использовать инструмент, ответь в формате:
<tool_call>
{{
  "tool": "название_инструмента",
  "parameters": {{
    "param1": "value1"
  }}
}}
</tool_call>

## РАБОЧИЙ ПРОЦЕСС:
1. Поприветствуй пользователя и спроси его имя
2. Узнай суть проблемы
3. Вызови create_ticket для регистрации обращения
4. Вызови search_knowledge_base для поиска информации
5. Сформулируй ответ на основе найденных данных
6. Сообщи пользователю номер созданного тикета

## ВАЖНЫЕ ПРАВИЛА:
- Будь вежлив и профессионален
- Задавай уточняющие вопросы если информации недостаточно
- Всегда создавай тикет перед поиском решения
- Основывай ответы на данных из базы знаний
'''
    
    return base_prompt.format(tools=tools_description)
```

### Функция `format_tools_description(tools: List[Dict]) -> str`
```
Реализуй форматирование описания инструментов:

def format_tools_description(tools: List[Dict[str, Any]]) -> str:
    """
    Форматирование списка инструментов для включения в промпт.
    """
    output = []
    
    for i, tool in enumerate(tools, 1):
        # Заголовок инструмента
        output.append(f"### {i}. {tool['name']}")
        output.append(f"**Описание:** {tool['description']}")
        output.append("**Параметры:**")
        
        # Параметры
        for param_name, param_info in tool.get('parameters', {}).items():
            required = "(required)" if param_info.get('required') else "(optional)"
            param_type = param_info.get('type', 'string')
            description = param_info.get('description', '')
            default = param_info.get('default')
            
            line = f"- {param_name} ({param_type}, {required}): {description}"
            if default is not None:
                line += f" [по умолчанию: {default}]"
            output.append(line)
        
        output.append("")  # Пустая строка между инструментами
    
    return "\n".join(output)
```

### Функция `get_tool_definitions() -> List[Dict]`
```
Реализуй получение определений инструментов:

def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Возвращает копию списка определений инструментов.
    """
    import copy
    return copy.deepcopy(TOOL_DEFINITIONS)
```

## Пример результата format_tools_description

Вход:
```python
tools = [
    {
        "name": "create_ticket",
        "description": "Создает тикет",
        "parameters": {
            "user_name": {"type": "string", "required": True, "description": "Имя"},
            "priority": {"type": "string", "required": False, "default": "medium"}
        }
    }
]
```

Выход:
```
### 1. create_ticket
**Описание:** Создает тикет
**Параметры:**
- user_name (string, (required)): Имя
- priority (string, (optional)):  [по умолчанию: medium]
```

## Зависимости
- copy (стандартная библиотека)
- typing

## Тестирование
После реализации проверь:
1. get_system_prompt() без аргументов возвращает SYSTEM_PROMPT
2. get_system_prompt(tools) возвращает промпт с кастомными инструментами
3. format_tools_description корректно форматирует все типы параметров
4. get_tool_definitions возвращает копию, а не ссылку
