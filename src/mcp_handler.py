"""
Обработчик MCP (Model Context Protocol) инструментов.

Обеспечивает вызов внешних инструментов через MCP серверы
и возврат результатов в LLM.

Поддерживает MCP Streamable HTTP транспорт с SSE (Server-Sent Events).
"""

import json
import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

import requests


@dataclass
class MCPServerConfig:
    """Конфигурация MCP сервера."""
    host: str
    port: int
    endpoint: str
    description: Optional[str] = None


@dataclass 
class ToolCallRequest:
    """Запрос на вызов инструмента."""
    tool_name: str
    parameters: Dict[str, Any]


@dataclass
class ToolCallResult:
    """Результат вызова инструмента."""
    success: bool
    result: Any
    error_message: Optional[str] = None


@dataclass
class MCPSession:
    """Сессия MCP соединения."""
    session_id: str
    server_name: str
    initialized: bool = False
    server_info: Dict[str, Any] = field(default_factory=dict)


class MCPHandler:
    """
    Обработчик MCP инструментов.
    
    Обеспечивает:
    - Регистрацию и управление MCP серверами
    - Вызов инструментов на удаленных серверах
    - Парсинг tool_call из ответов LLM
    - Форматирование результатов для LLM
    """
    
    def __init__(self, servers_config: Dict[str, MCPServerConfig]) -> None:
        """
        Инициализация обработчика.
        
        Args:
            servers_config: Словарь конфигураций серверов {name: config}
        """
        self._servers = servers_config
        self._local_tools: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, MCPSession] = {}  # server_name -> session
        self._request_id = 0
        
        # Маппинг инструментов на серверы (базовый, расширяется динамически)
        self._tool_to_server: Dict[str, Optional[str]] = {
            "search_knowledge_base": None,  # Локальный инструмент
        }
        # Остальные инструменты будут добавлены динамически при вызове get_server_tools()
    
    def _get_next_request_id(self) -> int:
        """Генерация уникального ID запроса."""
        self._request_id += 1
        return self._request_id
    
    def _parse_sse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Парсинг SSE ответа от MCP сервера.
        
        Args:
            response_text: Текст SSE ответа
            
        Returns:
            Распарсенный JSON или None
        """
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('data: '):
                json_str = line[6:]  # Remove 'data: ' prefix
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        return None
    
    def _initialize_session(self, server_name: str) -> MCPSession:
        """
        Инициализация сессии с MCP сервером.
        
        Args:
            server_name: Имя сервера
            
        Returns:
            Объект сессии
            
        Raises:
            MCPConnectionError: При ошибке подключения
        """
        server = self._servers.get(server_name)
        if not server:
            raise MCPConnectionError(f"Сервер '{server_name}' не найден в конфигурации")
        
        url = f"http://{server.host}:{server.port}{server.endpoint}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ticketManager",
                    "version": "1.0.0"
                }
            },
            "id": self._get_next_request_id()
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Явно устанавливаем кодировку UTF-8
            response.encoding = 'utf-8'
            
            # Получаем session_id из заголовков
            session_id = response.headers.get("mcp-session-id")
            if not session_id:
                raise MCPConnectionError("Сервер не вернул mcp-session-id")
            
            # Парсим SSE ответ
            result = self._parse_sse_response(response.text)
            if not result:
                raise MCPConnectionError("Не удалось распарсить ответ сервера")
            
            if "error" in result:
                raise MCPToolError(f"Ошибка инициализации: {result['error'].get('message', 'Unknown')}")
            
            session = MCPSession(
                session_id=session_id,
                server_name=server_name,
                initialized=True,
                server_info=result.get("result", {}).get("serverInfo", {})
            )
            
            self._sessions[server_name] = session
            return session
            
        except requests.exceptions.ConnectionError as e:
            raise MCPConnectionError(f"Не удалось подключиться к серверу {url}: {e}")
        except requests.exceptions.Timeout as e:
            raise MCPConnectionError(f"Таймаут подключения к серверу {url}: {e}")
        except requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Ошибка запроса к серверу {url}: {e}")
    
    def _get_or_create_session(self, server_name: str) -> MCPSession:
        """Получить существующую сессию или создать новую."""
        if server_name not in self._sessions:
            return self._initialize_session(server_name)
        return self._sessions[server_name]
    
    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """
        Вызов инструмента на MCP сервере или локально.
        
        Args:
            request: Запрос с именем инструмента и параметрами
            
        Returns:
            Результат выполнения инструмента
        """
        try:
            # Проверяем, есть ли инструмент в локальных
            if request.tool_name in self._local_tools:
                handler = self._local_tools[request.tool_name]["handler"]
                result = handler(**request.parameters)
                return ToolCallResult(success=True, result=result)
            
            # Иначе ищем сервер для инструмента
            server = self._get_server_for_tool(request.tool_name)
            
            if server is None:
                return ToolCallResult(
                    success=False, 
                    result=None,
                    error_message=f"Сервер для инструмента '{request.tool_name}' не найден"
                )
            
            # Отправляем запрос на MCP сервер
            result = self._send_mcp_request(server, request.tool_name, request.parameters)
            return ToolCallResult(success=True, result=result)
            
        except Exception as e:
            return ToolCallResult(success=False, result=None, error_message=str(e))
    
    def parse_tool_call(self, llm_response: str) -> Optional[ToolCallRequest]:
        """
        Парсинг вызова инструмента из ответа LLM.
        
        Args:
            llm_response: Текст ответа LLM
            
        Returns:
            ToolCallRequest если найден вызов, иначе None
        """
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        
        match = re.search(pattern, llm_response, re.DOTALL)
        
        if not match:
            return None
        
        try:
            json_content = match.group(1)
            data = json.loads(json_content)
            
            return ToolCallRequest(
                tool_name=data["tool"],
                parameters=data["parameters"]
            )
        except (json.JSONDecodeError, KeyError):
            return None
    
    def format_tool_result(self, tool_name: str, result: ToolCallResult) -> str:
        """
        Форматирование результата для отправки в LLM.
        
        Args:
            tool_name: Имя вызванного инструмента
            result: Результат выполнения
            
        Returns:
            Форматированная строка с результатом
        """
        if result.success:
            output = {
                "tool": tool_name,
                "success": True,
                "result": result.result
            }
        else:
            output = {
                "tool": tool_name,
                "success": False,
                "error": result.error_message
            }
        
        json_str = json.dumps(output, ensure_ascii=False, indent=2)
        return f"<tool_result>\n{json_str}\n</tool_result>"
    
    def has_tool_call(self, llm_response: str) -> bool:
        """
        Проверка наличия вызова инструмента в ответе.
        
        Args:
            llm_response: Текст ответа LLM
            
        Returns:
            True если найден <tool_call>, иначе False
        """
        return "<tool_call>" in llm_response and "</tool_call>" in llm_response
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Получение списка локальных инструментов.
        
        Для получения полного списка инструментов (включая MCP)
        используйте get_server_tools() для каждого сервера.
        
        Returns:
            Список описаний локальных инструментов
        """
        tools = []
        
        # Только локальные инструменты
        for name, tool_info in self._local_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "inputSchema": tool_info["parameters"]
            })
        
        return tools
    
    def register_local_tool(self, name: str, handler: Callable, 
                           description: str, parameters: Dict[str, Any]) -> None:
        """
        Регистрация локального инструмента (не MCP).
        
        Args:
            name: Имя инструмента
            handler: Функция-обработчик
            description: Описание инструмента
            parameters: Схема параметров
        """
        self._local_tools[name] = {
            "handler": handler,
            "description": description,
            "parameters": parameters
        }
    
    def _send_mcp_request(self, server: MCPServerConfig, 
                          tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправка запроса на MCP сервер с поддержкой Streamable HTTP.
        
        Args:
            server: Конфигурация сервера
            tool_name: Имя инструмента
            params: Параметры вызова
            
        Returns:
            Результат выполнения инструмента
            
        Raises:
            MCPConnectionError: При проблемах с подключением
            MCPToolError: При ошибке выполнения инструмента
        """
        # Находим имя сервера по конфигурации
        server_name = None
        for name, cfg in self._servers.items():
            if cfg.host == server.host and cfg.port == server.port:
                server_name = name
                break
        
        if not server_name:
            raise MCPConnectionError("Не удалось определить имя сервера")
        
        # Получаем или создаем сессию
        session = self._get_or_create_session(server_name)
        
        url = f"http://{server.host}:{server.port}{server.endpoint}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": self._get_next_request_id()
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session.session_id
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # Явно устанавливаем кодировку UTF-8 для корректной обработки кириллицы
            response.encoding = 'utf-8'
            
            # Парсим SSE ответ
            response_data = self._parse_sse_response(response.text)
            
            if not response_data:
                raise MCPToolError("Не удалось распарсить ответ сервера")
            
            if "error" in response_data:
                raise MCPToolError(
                    f"Ошибка инструмента: {response_data['error'].get('message', 'Unknown error')}"
                )
            
            return response_data["result"]
            
        except requests.exceptions.ConnectionError as e:
            # Сбрасываем сессию при ошибке соединения
            self._sessions.pop(server_name, None)
            raise MCPConnectionError(f"Не удалось подключиться к серверу {url}: {e}")
        except requests.exceptions.Timeout as e:
            raise MCPConnectionError(f"Таймаут подключения к серверу {url}: {e}")
        except requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Ошибка запроса к серверу {url}: {e}")
    
    def _get_server_for_tool(self, tool_name: str) -> Optional[MCPServerConfig]:
        """
        Определение сервера для инструмента.
        
        Args:
            tool_name: Имя инструмента
            
        Returns:
            Конфигурация сервера или None если инструмент локальный
        """
        server_name = self._tool_to_server.get(tool_name)
        
        if server_name is None:
            return None
        
        return self._servers.get(server_name)
    
    def check_servers_health(self) -> Dict[str, bool]:
        """
        Проверка доступности всех MCP серверов.
        
        Returns:
            Словарь {server_name: is_available}
        """
        health_status: Dict[str, bool] = {}
        
        for server_name, server_config in self._servers.items():
            try:
                # Пытаемся инициализировать сессию
                self._initialize_session(server_name)
                health_status[server_name] = True
            except (MCPConnectionError, MCPToolError):
                health_status[server_name] = False
        
        return health_status
    
    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Получение списка инструментов от MCP сервера.
        
        Args:
            server_name: Имя сервера
            
        Returns:
            Список инструментов сервера
        """
        server = self._servers.get(server_name)
        if not server:
            raise MCPConnectionError(f"Сервер '{server_name}' не найден")
        
        session = self._get_or_create_session(server_name)
        
        url = f"http://{server.host}:{server.port}{server.endpoint}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._get_next_request_id()
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session.session_id
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Явно устанавливаем кодировку UTF-8
            response.encoding = 'utf-8'
            
            response_data = self._parse_sse_response(response.text)
            
            if not response_data:
                raise MCPToolError("Не удалось распарсить ответ сервера")
            
            if "error" in response_data:
                raise MCPToolError(
                    f"Ошибка получения инструментов: {response_data['error'].get('message', 'Unknown')}"
                )
            
            tools = response_data.get("result", {}).get("tools", [])
            
            # Динамически регистрируем маппинг инструментов на сервер
            for tool in tools:
                tool_name = tool.get("name")
                if tool_name and tool_name not in self._tool_to_server:
                    self._tool_to_server[tool_name] = server_name
            
            return tools
            
        except requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Ошибка запроса: {e}")
    
    def close_session(self, server_name: str) -> None:
        """Закрытие сессии с сервером."""
        self._sessions.pop(server_name, None)


class MCPError(Exception):
    """Базовый класс ошибок MCP."""
    pass


class MCPConnectionError(MCPError):
    """Ошибка подключения к MCP серверу."""
    pass


class MCPToolError(MCPError):
    """Ошибка выполнения инструмента."""
    pass
