"""
Тесты для MCP Handler.

Включает unit-тесты и интеграционные тесты с реальным MCP сервером.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.mcp_handler import (
    MCPHandler,
    MCPServerConfig,
    ToolCallRequest,
    ToolCallResult,
    MCPSession,
    MCPConnectionError,
    MCPToolError,
)


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def server_config():
    """Конфигурация тестового сервера."""
    return {
        "ticket_service": MCPServerConfig(
            host="localhost",
            port=8000,
            endpoint="/mcp",
            description="Ticket service"
        )
    }


@pytest.fixture
def handler(server_config):
    """Экземпляр MCPHandler."""
    return MCPHandler(server_config)


@pytest.fixture
def live_handler():
    """MCPHandler для тестов с реальным сервером."""
    config = {
        "ticket_service": MCPServerConfig(
            host="localhost",
            port=8000,
            endpoint="/mcp",
            description="Live ticket service"
        )
    }
    return MCPHandler(config)


# ============================================================================
# Unit тесты: parse_tool_call
# ============================================================================

class TestParseToolCall:
    """Тесты парсинга tool_call из ответа LLM."""
    
    def test_parse_valid_tool_call(self, handler):
        """Парсинг корректного вызова инструмента."""
        response = '''
Хорошо, я создам тикет для вас.

<tool_call>
{
  "tool": "create_ticket",
  "parameters": {
    "author": "Иван Петров",
    "theme": "Проблема со входом",
    "description": "Не работает вход в систему"
  }
}
</tool_call>
'''
        result = handler.parse_tool_call(response)
        
        assert result is not None
        assert result.tool_name == "create_ticket"
        assert result.parameters["author"] == "Иван Петров"
        assert result.parameters["theme"] == "Проблема со входом"
        assert result.parameters["description"] == "Не работает вход в систему"
    
    def test_parse_tool_call_no_tag(self, handler):
        """Возвращает None если нет тега tool_call."""
        response = "Привет! Чем могу помочь?"
        result = handler.parse_tool_call(response)
        assert result is None
    
    def test_parse_tool_call_invalid_json(self, handler):
        """Возвращает None при невалидном JSON."""
        response = '''
<tool_call>
{invalid json}
</tool_call>
'''
        result = handler.parse_tool_call(response)
        assert result is None
    
    def test_parse_tool_call_missing_fields(self, handler):
        """Возвращает None при отсутствии обязательных полей."""
        response = '''
<tool_call>
{
  "name": "create_ticket"
}
</tool_call>
'''
        result = handler.parse_tool_call(response)
        assert result is None
    
    def test_parse_tool_call_multiline(self, handler):
        """Парсинг многострочного JSON."""
        response = '''<tool_call>
{
    "tool": "get_ticket_status",
    "parameters": {
        "ticket_id": "TKT-12345"
    }
}
</tool_call>'''
        result = handler.parse_tool_call(response)
        
        assert result is not None
        assert result.tool_name == "get_ticket_status"
        assert result.parameters["ticket_id"] == "TKT-12345"


# ============================================================================
# Unit тесты: has_tool_call
# ============================================================================

class TestHasToolCall:
    """Тесты проверки наличия tool_call."""
    
    def test_has_tool_call_present(self, handler):
        """Возвращает True когда тег присутствует."""
        response = "Текст <tool_call>{}</tool_call> текст"
        assert handler.has_tool_call(response) is True
    
    def test_has_tool_call_absent(self, handler):
        """Возвращает False когда тега нет."""
        response = "Обычный текст без инструментов"
        assert handler.has_tool_call(response) is False
    
    def test_has_tool_call_only_opening(self, handler):
        """Возвращает False если только открывающий тег."""
        response = "Текст <tool_call> без закрытия"
        assert handler.has_tool_call(response) is False
    
    def test_has_tool_call_only_closing(self, handler):
        """Возвращает False если только закрывающий тег."""
        response = "Текст </tool_call> без открытия"
        assert handler.has_tool_call(response) is False


# ============================================================================
# Unit тесты: format_tool_result
# ============================================================================

class TestFormatToolResult:
    """Тесты форматирования результата."""
    
    def test_format_success_result(self, handler):
        """Форматирование успешного результата."""
        result = ToolCallResult(
            success=True,
            result={"ticket_id": "TKT-123", "status": "created"}
        )
        
        formatted = handler.format_tool_result("create_ticket", result)
        
        assert "<tool_result>" in formatted
        assert "</tool_result>" in formatted
        assert '"success": true' in formatted
        assert '"ticket_id": "TKT-123"' in formatted
        assert "create_ticket" in formatted
    
    def test_format_error_result(self, handler):
        """Форматирование результата с ошибкой."""
        result = ToolCallResult(
            success=False,
            result=None,
            error_message="Сервер недоступен"
        )
        
        formatted = handler.format_tool_result("create_ticket", result)
        
        assert "<tool_result>" in formatted
        assert "</tool_result>" in formatted
        assert '"success": false' in formatted
        assert "Сервер недоступен" in formatted
    
    def test_format_result_cyrillic(self, handler):
        """Кириллица корректно обрабатывается."""
        result = ToolCallResult(
            success=True,
            result={"message": "Тикет создан успешно"}
        )
        
        formatted = handler.format_tool_result("create_ticket", result)
        
        assert "Тикет создан успешно" in formatted


# ============================================================================
# Unit тесты: register_local_tool
# ============================================================================

class TestRegisterLocalTool:
    """Тесты регистрации локальных инструментов."""
    
    def test_register_tool(self, handler):
        """Регистрация локального инструмента."""
        def dummy_handler(query: str):
            return {"results": [query]}
        
        handler.register_local_tool(
            name="search_kb",
            handler=dummy_handler,
            description="Поиск в базе знаний",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}}
        )
        
        assert "search_kb" in handler._local_tools
        assert handler._local_tools["search_kb"]["description"] == "Поиск в базе знаний"
    
    def test_call_local_tool(self, handler):
        """Вызов зарегистрированного локального инструмента."""
        def search_handler(query: str):
            return {"found": True, "query": query}
        
        handler.register_local_tool(
            name="test_search",
            handler=search_handler,
            description="Test search",
            parameters={}
        )
        
        request = ToolCallRequest(
            tool_name="test_search",
            parameters={"query": "тест"}
        )
        
        result = handler.call_tool(request)
        
        assert result.success is True
        assert result.result["found"] is True
        assert result.result["query"] == "тест"


# ============================================================================
# Unit тесты: get_available_tools
# ============================================================================

class TestGetAvailableTools:
    """Тесты получения списка инструментов."""
    
    def test_get_mcp_tools(self, handler):
        """Получение MCP инструментов."""
        tools = handler.get_available_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "create_ticket" in tool_names
        assert "get_ticket" in tool_names
        assert "list_tickets" in tool_names
        assert "update_ticket" in tool_names
        assert "delete_ticket" in tool_names
    
    def test_get_tools_includes_local(self, handler):
        """Локальные инструменты включены в список."""
        handler.register_local_tool(
            name="local_tool",
            handler=lambda: None,
            description="Локальный инструмент",
            parameters={}
        )
        
        tools = handler.get_available_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "local_tool" in tool_names


# ============================================================================
# Unit тесты: _parse_sse_response
# ============================================================================

class TestParseSSEResponse:
    """Тесты парсинга SSE ответов."""
    
    def test_parse_valid_sse(self, handler):
        """Парсинг валидного SSE ответа."""
        sse_response = '''event: message
data: {"jsonrpc":"2.0","id":1,"result":{"status":"ok"}}

'''
        result = handler._parse_sse_response(sse_response)
        
        assert result is not None
        assert result["jsonrpc"] == "2.0"
        assert result["result"]["status"] == "ok"
    
    def test_parse_sse_without_event(self, handler):
        """Парсинг SSE без строки event."""
        sse_response = 'data: {"jsonrpc":"2.0","id":1,"result":{}}'
        result = handler._parse_sse_response(sse_response)
        
        assert result is not None
        assert result["jsonrpc"] == "2.0"
    
    def test_parse_sse_empty(self, handler):
        """Пустой ответ возвращает None."""
        result = handler._parse_sse_response("")
        assert result is None
    
    def test_parse_sse_no_data_line(self, handler):
        """Ответ без data: возвращает None."""
        result = handler._parse_sse_response("event: message\n\n")
        assert result is None


# ============================================================================
# Интеграционные тесты с реальным MCP сервером
# ============================================================================

class TestMCPServerIntegration:
    """
    Интеграционные тесты с реальным MCP сервером.
    
    Требуют запущенного MCP сервера на localhost:8000.
    """
    
    @pytest.fixture(autouse=True)
    def check_server(self, live_handler):
        """Проверка доступности сервера перед тестами."""
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", 
                      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "test", "version": "1.0"}}, "id": 1},
                headers={"Accept": "application/json, text/event-stream"},
                timeout=5
            )
            if response.status_code != 200:
                pytest.skip("MCP сервер недоступен")
        except requests.exceptions.RequestException:
            pytest.skip("MCP сервер недоступен")
    
    def test_initialize_session(self, live_handler):
        """Инициализация сессии с сервером."""
        session = live_handler._initialize_session("ticket_service")
        
        assert session is not None
        assert session.session_id is not None
        assert session.initialized is True
        assert "ticket_service" in live_handler._sessions
    
    def test_get_server_tools(self, live_handler):
        """Получение списка инструментов от сервера."""
        tools = live_handler.get_server_tools("ticket_service")
        
        assert isinstance(tools, list)
        # Проверяем что хотя бы какие-то инструменты есть
        if tools:
            assert "name" in tools[0]
            print(f"\nДоступные инструменты: {[t['name'] for t in tools]}")
    
    def test_check_servers_health(self, live_handler):
        """Проверка здоровья серверов."""
        health = live_handler.check_servers_health()
        
        assert "ticket_service" in health
        assert health["ticket_service"] is True
    
    def test_call_create_ticket(self, live_handler):
        """Вызов инструмента create_ticket."""
        request = ToolCallRequest(
            tool_name="create_ticket",
            parameters={
                "author": "Тест Пользователь",
                "theme": "Тестовый тикет",
                "description": "Тестовый тикет из pytest"
            }
        )
        
        result = live_handler.call_tool(request)
        
        print(f"\nРезультат create_ticket: {result}")
        
        assert result.success is True
        assert result.result is not None
        # Проверяем что нет ошибки в результате
        if isinstance(result.result, dict) and "isError" in result.result:
            assert result.result.get("isError") is not True, f"Tool error: {result.result}"
    
    def test_call_list_tickets(self, live_handler):
        """Вызов инструмента list_tickets."""
        request = ToolCallRequest(
            tool_name="list_tickets",
            parameters={}
        )
        
        result = live_handler.call_tool(request)
        
        print(f"\nРезультат list_tickets: {result}")
        
        assert result.success is True
        assert result.result is not None
    
    def test_call_get_ticket(self, live_handler):
        """Вызов инструмента get_ticket после создания."""
        # Сначала создаем тикет
        create_request = ToolCallRequest(
            tool_name="create_ticket",
            parameters={
                "author": "Тест",
                "theme": "Тема для get_ticket",
                "description": "Тикет для проверки get_ticket"
            }
        )
        create_result = live_handler.call_tool(create_request)
        
        if not create_result.success:
            pytest.skip("Не удалось создать тикет для теста")
        
        # Извлекаем id из результата
        ticket_id = None
        if isinstance(create_result.result, dict):
            ticket_id = create_result.result.get("id")
            if not ticket_id and "content" in create_result.result:
                # Результат в формате MCP content
                content = create_result.result.get("content", [])
                if content and isinstance(content[0], dict):
                    text = content[0].get("text", "")
                    # Пробуем извлечь UUID из текста
                    import re
                    # UUID pattern
                    match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text, re.I)
                    if match:
                        ticket_id = match.group(0)
        
        if not ticket_id:
            print(f"\nСоздан тикет (результат): {create_result.result}")
            pytest.skip("Не удалось извлечь ticket id")
        
        # Получаем тикет
        get_request = ToolCallRequest(
            tool_name="get_ticket",
            parameters={"id": ticket_id}
        )
        
        get_result = live_handler.call_tool(get_request)
        
        print(f"\nТикет {ticket_id}: {get_result}")
        
        assert get_result.success is True
    
    def test_session_reuse(self, live_handler):
        """Повторное использование сессии."""
        # Первый вызов - создает сессию
        live_handler._get_or_create_session("ticket_service")
        session_id_1 = live_handler._sessions["ticket_service"].session_id
        
        # Второй вызов - использует существующую
        live_handler._get_or_create_session("ticket_service")
        session_id_2 = live_handler._sessions["ticket_service"].session_id
        
        assert session_id_1 == session_id_2
    
    def test_full_workflow(self, live_handler):
        """
        Полный рабочий процесс:
        1. LLM генерирует ответ с tool_call
        2. Парсим tool_call
        3. Выполняем инструмент
        4. Форматируем результат
        """
        # Симулируем ответ LLM
        llm_response = '''
Хорошо, я создам для вас тикет.

<tool_call>
{
  "tool": "create_ticket",
  "parameters": {
    "author": "Интеграционный Тест",
    "theme": "Полный рабочий процесс",
    "description": "Тестирование полного рабочего процесса MCP"
  }
}
</tool_call>
'''
        
        # 1. Проверяем наличие tool_call
        assert live_handler.has_tool_call(llm_response) is True
        
        # 2. Парсим
        request = live_handler.parse_tool_call(llm_response)
        assert request is not None
        assert request.tool_name == "create_ticket"
        
        # 3. Выполняем
        result = live_handler.call_tool(request)
        print(f"\nРезультат выполнения: {result}")
        
        # Проверяем успешность
        assert result.success is True
        if isinstance(result.result, dict) and "isError" in result.result:
            assert result.result.get("isError") is not True, f"Tool error: {result.result}"
        
        # 4. Форматируем
        formatted = live_handler.format_tool_result(request.tool_name, result)
        print(f"\nОтформатированный результат:\n{formatted}")
        
        assert "<tool_result>" in formatted
        assert "</tool_result>" in formatted


# ============================================================================
# Тесты обработки ошибок
# ============================================================================

class TestErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_call_unknown_tool(self, handler):
        """Вызов несуществующего инструмента."""
        request = ToolCallRequest(
            tool_name="unknown_tool",
            parameters={}
        )
        
        result = handler.call_tool(request)
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_server_not_found(self):
        """Сервер не найден в конфигурации."""
        handler = MCPHandler({})
        
        with pytest.raises(MCPConnectionError):
            handler._initialize_session("nonexistent_server")
    
    def test_connection_error_handling(self):
        """Обработка ошибки подключения."""
        config = {
            "bad_server": MCPServerConfig(
                host="192.0.2.1",  # Недоступный адрес
                port=9999,
                endpoint="/mcp"
            )
        }
        handler = MCPHandler(config)
        
        # Маппим инструмент на плохой сервер
        handler._tool_to_server["test_tool"] = "bad_server"
        
        request = ToolCallRequest(
            tool_name="test_tool",
            parameters={}
        )
        
        result = handler.call_tool(request)
        
        assert result.success is False
        assert "подключиться" in result.error_message.lower() or "connect" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
