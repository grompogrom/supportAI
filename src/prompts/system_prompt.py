"""
Системный промпт для LLM ассистента поддержки.

Содержит:
- Роль и обязанности ассистента
- Описание доступных инструментов
- Формат вызова инструментов
- Рабочий процесс
"""

import copy
from typing import List, Dict, Any


# Основной системный промпт
SYSTEM_PROMPT = """
Ты - ассистент службы поддержки. Твоя задача - помогать пользователям решать их проблемы.

## ТВОИ ОБЯЗАННОСТИ:
1. Приветствовать пользователя и узнать его имя
2. Выяснить суть проблемы или вопроса
3. Создать тикет в системе поддержки
4. Найти релевантную информацию в базе знаний
5. Предоставить пользователю полезный ответ

## ДОСТУПНЫЕ ИНСТРУМЕНТЫ (MCP TOOLS):

### 1. create_ticket
**Описание:** Создает тикет в системе поддержки
**Параметры:**
- user_name (string, required): Имя пользователя
- issue_summary (string, required): Краткое описание проблемы
- issue_details (string, required): Подробное описание проблемы
- priority (string, optional): low/medium/high, по умолчанию medium

**Возвращает:** ticket_id, статус создания

### 2. search_knowledge_base
**Описание:** Поиск информации в базе знаний (RAG)
**Параметры:**
- query (string, required): Поисковый запрос

**Возвращает:** Список релевантных фрагментов документации

### 3. get_ticket_status
**Описание:** Получить статус тикета
**Параметры:**
- ticket_id (string, required): ID тикета

**Возвращает:** Текущий статус тикета

## ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:
Когда тебе нужно использовать инструмент, ответь в формате:
<tool_call>
{
  "tool": "название_инструмента",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
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
- Если не можешь найти ответ - честно сообщи об этом и предложи связаться с живым оператором
- Не выдумывай информацию, которой нет в базе знаний
"""


def get_system_prompt(tools_override: List[Dict[str, Any]] = None) -> str:
    """
    Получение системного промпта с возможностью кастомизации.
    
    Args:
        tools_override: Опциональный список MCP инструментов для добавления к базовым
        
    Returns:
        Готовый системный промпт
        
    Действия:
    - Если tools_override указан - добавить MCP инструменты к встроенному search_knowledge_base
    - Подставить в шаблон промпта
    - Вернуть финальный промпт
    """
    if tools_override is None:
        return SYSTEM_PROMPT
    
    # Сгенерировать описание MCP инструментов
    mcp_tools_description = format_tools_description(tools_override)
    
    # Базовый промпт со статическим search_knowledge_base и динамическими MCP инструментами
    base_prompt = '''Ты - ассистент службы поддержки. Твоя задача - помогать пользователям решать их проблемы.

## ТВОИ ОБЯЗАННОСТИ:
1. Приветствовать пользователя и узнать его имя
2. Выяснить суть проблемы или вопроса
3. Создать тикет в системе поддержки
4. Найти релевантную информацию в базе знаний
5. Предоставить пользователю полезный ответ

## ДОСТУПНЫЕ ИНСТРУМЕНТЫ:

### Встроенный инструмент (RAG):

**search_knowledge_base**
- Описание: Поиск информации в базе знаний (RAG)
- Параметры:
  - query (string, required): Поисковый запрос
- Возвращает: Список релевантных фрагментов документации

### MCP инструменты (внешние):

{mcp_tools}

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

**КРИТИЧЕСКИ ВАЖНО:** Используй ТОЧНЫЕ названия параметров из описания инструментов выше! 
Для MCP инструментов используй параметры, указанные в их описании.

## РАБОЧИЙ ПРОЦЕСС:
1. Поприветствуй пользователя и спроси его имя
2. Узнай суть проблемы
3. Вызови create_ticket для регистрации обращения (используй параметры из описания MCP инструмента)
4. Вызови search_knowledge_base для поиска информации в базе знаний
5. Сформулируй ответ на основе найденных данных
6. Сообщи пользователю номер созданного тикета

## ВАЖНЫЕ ПРАВИЛА:
- Будь вежлив и профессионален
- Задавай уточняющие вопросы если информации недостаточно
- Всегда создавай тикет перед поиском решения
- Основывай ответы на данных из базы знаний
- Используй ТОЛЬКО параметры, указанные в описании инструментов
- Если не можешь найти ответ - честно сообщи об этом и предложи связаться с живым оператором
- Не выдумывай информацию, которой нет в базе знаний
'''
    
    return base_prompt.format(mcp_tools=mcp_tools_description)


def format_tools_description(tools: List[Dict[str, Any]]) -> str:
    """
    Форматирование описания инструментов для промпта.
    
    Поддерживает два формата:
    1. Локальный формат:
       {"name": "...", "description": "...", "parameters": {"param1": {"type": "...", "required": True}}}
    2. MCP JSON Schema формат:
       {"name": "...", "description": "...", "inputSchema": {"type": "object", "properties": {...}, "required": [...]}}
    
    Args:
        tools: Список описаний инструментов
        
    Returns:
        Форматированная строка с описанием всех инструментов
    """
    output = []
    
    for i, tool in enumerate(tools, 1):
        # Заголовок инструмента
        output.append(f"### {i}. {tool['name']}")
        output.append(f"**Описание:** {tool.get('description', 'Без описания')}")
        output.append("**Параметры:**")
        
        # Определяем формат параметров
        if 'inputSchema' in tool:
            # MCP JSON Schema формат
            schema = tool['inputSchema']
            properties = schema.get('properties', {})
            required_params = schema.get('required', [])
            
            if not properties:
                output.append("- Нет параметров")
            else:
                for param_name, param_info in properties.items():
                    is_required = param_name in required_params
                    required_str = "(required)" if is_required else "(optional)"
                    param_type = param_info.get('type', 'string')
                    description = param_info.get('description', '')
                    default = param_info.get('default')
                    
                    line = f"- {param_name} ({param_type}, {required_str}): {description}"
                    if default is not None:
                        line += f" [по умолчанию: {default}]"
                    output.append(line)
        else:
            # Локальный формат или старый формат
            params = tool.get('parameters', {})
            
            # Проверяем, является ли это JSON Schema в старом поле parameters
            if params.get('type') == 'object' and 'properties' in params:
                properties = params.get('properties', {})
                required_params = params.get('required', [])
                
                if not properties:
                    output.append("- Нет параметров")
                else:
                    for param_name, param_info in properties.items():
                        is_required = param_name in required_params
                        required_str = "(required)" if is_required else "(optional)"
                        param_type = param_info.get('type', 'string')
                        description = param_info.get('description', '')
                        default = param_info.get('default')
                        
                        line = f"- {param_name} ({param_type}, {required_str}): {description}"
                        if default is not None:
                            line += f" [по умолчанию: {default}]"
                        output.append(line)
            elif not params:
                output.append("- Нет параметров")
            else:
                # Старый локальный формат с прямыми параметрами
                for param_name, param_info in params.items():
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


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Получение определений всех доступных инструментов.
    
    Returns:
        Список словарей с описаниями инструментов
        
    Используется для:
    - Генерации динамического промпта
    - Валидации вызовов инструментов
    - Документации API
    """
    return copy.deepcopy(TOOL_DEFINITIONS)


# Определения инструментов в структурированном формате
TOOL_DEFINITIONS = [
    {
        "name": "create_ticket",
        "description": "Создает тикет в системе поддержки",
        "parameters": {
            "user_name": {
                "type": "string",
                "required": True,
                "description": "Имя пользователя"
            },
            "issue_summary": {
                "type": "string", 
                "required": True,
                "description": "Краткое описание проблемы"
            },
            "issue_details": {
                "type": "string",
                "required": True,
                "description": "Подробное описание проблемы"
            },
            "priority": {
                "type": "string",
                "required": False,
                "description": "Приоритет: low/medium/high",
                "default": "medium"
            }
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Поиск информации в базе знаний (RAG)",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Поисковый запрос"
            }
        }
    },
    {
        "name": "get_ticket_status",
        "description": "Получить статус тикета",
        "parameters": {
            "ticket_id": {
                "type": "string",
                "required": True,
                "description": "ID тикета"
            }
        }
    }
]
