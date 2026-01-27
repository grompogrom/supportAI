"""
Системный промпт для LLM менеджера проекта.

Содержит:
- Роль и обязанности AI менеджера проекта
- Описание доступных инструментов (MCP + встроенные)
- Формат вызова инструментов
- Типичные сценарии работы
- Правила управления проектом

Системный промпт загружается из файла system_prompt.txt
"""

import os
from typing import List, Dict, Any


def _load_prompt_from_file(filename: str) -> str:
    """
    Загрузка системного промпта из текстового файла.
    
    Args:
        filename: Имя файла относительно директории prompts
        
    Returns:
        Содержимое файла как строка
        
    Raises:
        FileNotFoundError: Если файл не найден
        IOError: Если не удалось прочитать файл
    """
    # Получаем путь к директории prompts
    prompts_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(prompts_dir, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл системного промпта не найден: {file_path}")
    except IOError as e:
        raise IOError(f"Ошибка при чтении файла системного промпта: {e}")


# Основной системный промпт (загружается из файла)
SYSTEM_PROMPT = _load_prompt_from_file("system_prompt.txt")


def get_system_prompt(tools_override: List[Dict[str, Any]] = None) -> str:
    """
    Получение системного промпта для AI менеджера проекта с возможностью кастомизации.
    
    Args:
        tools_override: Опциональный список MCP инструментов для добавления к встроенным
        
    Returns:
        Готовый системный промпт для AI менеджера проекта
        
    Действия:
    - Всегда использует базовый промпт из system_prompt.txt
    - Если tools_override указан - добавляет описание MCP инструментов
    - Вернуть финальный промпт с инструкциями по управлению проектом
    """
    # Всегда используем базовый промпт из system_prompt.txt
    base_prompt = SYSTEM_PROMPT
    
    if tools_override is None or len(tools_override) == 0:
        return base_prompt
    
    # Сгенерировать описание MCP инструментов
    mcp_tools_description = format_tools_description(tools_override)
    
    # Найти и заменить секцию MCP инструментов в базовом промпте
    # Ищем маркер начала секции MCP инструментов
    mcp_section_start = "### MCP инструменты"
    
    if mcp_section_start in base_prompt:
        # Находим начало секции MCP инструментов
        start_idx = base_prompt.find(mcp_section_start)
        
        # Находим конец секции MCP инструментов (следующий заголовок ##)
        # Ищем следующий заголовок ## (но не ###)
        remaining = base_prompt[start_idx:]
        next_section_idx = None
        
        # Ищем следующий заголовок ## после секции MCP
        for i, line in enumerate(remaining.split('\n')[1:], 1):  # Пропускаем заголовок секции
            stripped = line.strip()
            if stripped.startswith('## ') and not stripped.startswith('###'):
                # Находим позицию этого заголовка в исходной строке
                lines_before = remaining.split('\n')[:i]
                next_section_idx = start_idx + len('\n'.join(lines_before))
                break
        
        if next_section_idx:
            # Заменяем секцию MCP инструментов
            before = base_prompt[:start_idx + len(mcp_section_start)]
            # Находим конец строки заголовка
            header_end = base_prompt.find('\n', start_idx)
            if header_end == -1:
                header_end = len(base_prompt)
            before = base_prompt[:header_end + 1]  # Включаем перенос строки
            after = base_prompt[next_section_idx:]
            return before + "\n" + mcp_tools_description + "\n\n" + after
        else:
            # Если не нашли следующий раздел, заменяем до конца
            header_end = base_prompt.find('\n', start_idx)
            if header_end == -1:
                header_end = len(base_prompt)
            before = base_prompt[:header_end + 1]
            return before + "\n" + mcp_tools_description
    
    # Если секция MCP инструментов не найдена в базовом промпте,
    # просто добавляем описание инструментов в конец
    return base_prompt + "\n\n## ДОПОЛНИТЕЛЬНЫЕ ИНСТРУМЕНТЫ:\n\n" + mcp_tools_description


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


