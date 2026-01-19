"""
Главный модуль приложения Support Assistant.

Содержит точку входа и консольный интерфейс для взаимодействия
с пользователем.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Any, Dict, List

import yaml

from llm_client import PerplexityClient, LocalLLMClient, BaseLLMClient
from mcp_handler import MCPHandler, MCPServerConfig, ToolCallRequest
from rag import DocumentIndexer, EmbeddingGenerator, DocumentRetriever
from rag.embeddings import EmbeddingConfig
from prompts import get_system_prompt


def load_config(config_path: str) -> dict:
    """
    Загрузка конфигурации из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с конфигурацией
        
    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если ошибка парсинга YAML
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SupportAssistant:
    """
    Основной класс ассистента поддержки.
    
    Координирует работу всех компонентов:
    - LLM клиент для генерации ответов
    - MCP обработчик для вызова инструментов
    - RAG система для поиска в документации
    """
    
    def __init__(self) -> None:
        """
        Инициализация ассистента.
        
        Действия:
        - Загрузить все конфигурации
        - Инициализировать LLM клиент
        - Инициализировать MCP обработчик
        - Загрузить системный промпт
        - Инициализировать историю диалога
        """
        # Определи пути к конфигам (относительно src/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Загрузи конфигурации
        self._api_config = load_config(os.path.join(base_dir, 'config', 'api_keys.yaml'))
        self._mcp_config = load_config(os.path.join(base_dir, 'config', 'mcp_config.yaml'))
        self._llm_config = load_config(os.path.join(base_dir, 'config', 'local_llm_config.yaml'))
        
        # Пути к данным
        self._docs_dir = os.path.join(base_dir, 'docs')
        self._embeddings_path = os.path.join(base_dir, 'data', 'embeddings.json')
        
        # 1. Embedding Generator
        emb_config = EmbeddingConfig(
            host=self._llm_config['embedding_model']['host'],
            port=self._llm_config['embedding_model']['port'],
            model_name=self._llm_config['embedding_model']['model_name'],
            endpoint=self._llm_config['embedding_model']['endpoint']
        )
        self._embedding_generator = EmbeddingGenerator(emb_config)
        
        # 2. Document Indexer
        self._indexer = DocumentIndexer(
            self._docs_dir,
            self._embeddings_path
        )
        
        # 3. Document Retriever
        self._retriever = DocumentRetriever(
            self._embeddings_path,
            self._embedding_generator
        )
        
        # 4. MCP Handler
        servers = {}
        for name, cfg in self._mcp_config['servers'].items():
            servers[name] = MCPServerConfig(
                host=cfg['host'],
                port=cfg['port'],
                endpoint=cfg['endpoint']
            )
        self._mcp_handler = MCPHandler(servers)
        
        # Зарегистрируй локальный инструмент search_knowledge_base
        self._mcp_handler.register_local_tool(
            name="search_knowledge_base",
            handler=self._search_knowledge_base,
            description="Поиск в базе знаний",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    }
                },
                "required": ["query"]
            }
        )

        # Зарегистрируй локальный инструмент recommend_tasks
        self._mcp_handler.register_local_tool(
            name="recommend_tasks",
            handler=self._recommend_tasks,
            description="Детерминированные рекомендации по приоритету задач",
            parameters={
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Фильтр по приоритету (например: critical/high/medium/low)"
                    },
                    "status": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Фильтр по статусу (например: open/in_progress/blocked/done)"
                    }
                }
            }
        )
        
        # 5. Получение списка инструментов от MCP сервера
        tools = self._fetch_mcp_tools()
        
        # 6. LLM Client с динамическим системным промптом
        system_prompt = get_system_prompt(tools_override=tools)
        self._llm_client = self._create_llm_client(system_prompt)
    
    def _fetch_mcp_tools(self) -> list:
        """
        Получение списка инструментов от MCP серверов.
        
        Returns:
            Список инструментов с описаниями и схемами параметров
        """
        tools = []
        
        # Получаем инструменты от каждого MCP сервера
        for server_name in self._mcp_handler._servers.keys():
            try:
                server_tools = self._mcp_handler.get_server_tools(server_name)
                tools.extend(server_tools)
                print(f"[MCP] Загружено {len(server_tools)} инструментов от {server_name}")
            except Exception as e:
                print(f"[MCP] Предупреждение: не удалось получить инструменты от {server_name}: {e}")
        
        # Добавляем локальные инструменты
        for name, tool_info in self._mcp_handler._local_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "inputSchema": tool_info["parameters"]
            })
        
        return tools
    
    def _create_llm_client(self, system_prompt: str) -> BaseLLMClient:
        """
        Создание LLM клиента в зависимости от конфигурации.
        
        Args:
            system_prompt: Системный промпт для модели
            
        Returns:
            Экземпляр LLM клиента (Perplexity или локальный)
        """
        provider = self._api_config.get('llm_provider', 'perplexity').lower()
        
        if provider == 'local':
            # Используем локальную модель через Ollama
            chat_config = self._llm_config.get('chat_model', {})
            host = chat_config.get('host', 'localhost')
            port = chat_config.get('port', 11434)
            model_name = chat_config.get('model_name', 'qwen3:8b')
            temperature = chat_config.get('temperature', 0.7)
            
            print(f"[LLM] Используется локальная модель: {model_name} на {host}:{port}")
            
            client = LocalLLMClient(
                host=host,
                port=port,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            # Проверяем доступность модели
            if not client.check_model_availability():
                print(f"[LLM] ПРЕДУПРЕЖДЕНИЕ: Модель {model_name} недоступна!")
                print(f"[LLM] Убедитесь, что Ollama запущен и модель загружена:")
                print(f"[LLM]   ollama run {model_name}")
            
            return client
        else:
            # Используем Perplexity API (по умолчанию)
            api_key = self._api_config['perplexity']['api_key']
            print(f"[LLM] Используется Perplexity API (sonar-pro)")
            
            return PerplexityClient(
                api_key=api_key,
                system_prompt=system_prompt
            )
    
    def start(self) -> None:
        """
        Запуск консольного интерфейса.
        
        Действия:
        - Вывести приветственное сообщение
        - Запустить главный цикл обработки ввода
        - Обрабатывать команды и сообщения пользователя
        """
        self.print_welcome()
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                response = self.process_input(user_input)
                
                if response:
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n\nВыход из программы...")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")
    
    def process_input(self, user_input: str) -> Optional[str]:
        """
        Обработка ввода пользователя.
        
        Args:
            user_input: Текст, введенный пользователем
            
        Returns:
            Ответ ассистента или None для команд без ответа
            
        Действия:
        - Проверить, является ли ввод командой (начинается с /)
        - Если команда - вызвать соответствующий обработчик
        - Если сообщение - отправить в LLM и обработать ответ
        """
        # Проверь команды
        if user_input.startswith('/'):
            return self.handle_command(user_input)
        
        # Отправь сообщение в LLM
        return self.send_to_llm(user_input)
    
    def handle_command(self, command: str) -> Optional[str]:
        """
        Обработка команд пользователя.
        
        Args:
            command: Команда (например, /index, /clear, /exit)
            
        Returns:
            Результат выполнения команды или None
            
        Поддерживаемые команды:
        - /index - запуск индексации документов
        - /clear - очистка истории диалога
        - /status <ticket_id> - проверка статуса тикета
        - /help - показать справку по командам
        - /exit или /quit - выход из программы
        """
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == '/index':
            return self._do_index()
        elif cmd == '/clear':
            self.clear_history()
            return "История диалога очищена."
        elif cmd == '/status' and args:
            return self._check_ticket_status(args[0])
        elif cmd == '/help':
            self.print_help()
            return None
        elif cmd in ['/exit', '/quit']:
            print("До свидания!")
            sys.exit(0)
        else:
            return f"Неизвестная команда: {cmd}. Введите /help для справки."
    
    def _do_index(self) -> str:
        """Запуск индексации документов."""
        print("Начинаю индексацию документов...")
        try:
            result = self._indexer.index_all(self._embedding_generator)
            return (f"Индексация завершена!\n"
                    f"Файлов: {result.total_files}\n"
                    f"Чанков: {result.total_chunks}\n"
                    f"Ошибок: {len(result.errors)}")
        except Exception as e:
            return f"Ошибка индексации: {e}"
    
    def _check_ticket_status(self, ticket_id: str) -> str:
        """Проверка статуса тикета через MCP."""
        request = ToolCallRequest(
            tool_name="get_ticket",
            parameters={"id": ticket_id}
        )
        result = self._mcp_handler.call_tool(request)
        if result.success:
            ticket_data = result.result
            # Форматируем вывод тикета
            if isinstance(ticket_data, dict):
                content = ticket_data.get('content', [])
                if content and isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            return f"Тикет {ticket_id}:\n{item.get('text', 'Нет данных')}"
            return f"Тикет {ticket_id}: {ticket_data}"
        return f"Ошибка: {result.error_message}"
    
    def send_to_llm(self, message: str) -> str:
        """
        Отправка сообщения в LLM и получение ответа.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Ответ от LLM (возможно после обработки tool calls)
            
        Действия:
        - Добавить сообщение в историю
        - Отправить запрос в Perplexity API
        - Проверить ответ на наличие tool_call
        - При наличии tool_call - выполнить и отправить результат обратно
        - Вернуть финальный ответ
        """
        # Отправь сообщение
        response = self._llm_client.send_message(message)
        
        # Проверь и обработай tool calls
        return self.process_tool_calls(response)
    
    def process_tool_calls(self, llm_response: str) -> str:
        """
        Обработка вызовов инструментов из ответа LLM.
        
        Args:
            llm_response: Ответ LLM, возможно содержащий <tool_call>
            
        Returns:
            Финальный ответ после выполнения всех tool calls
            
        Действия:
        - Распарсить ответ на наличие <tool_call> блоков
        - Извлечь имя инструмента и параметры
        - Вызвать инструмент через MCP handler
        - Отправить результат обратно в LLM
        - Повторить при необходимости
        """
        # Максимум 5 итераций для защиты от бесконечного цикла
        max_iterations = 5
        current_response = llm_response
        
        for _ in range(max_iterations):
            # Проверь наличие tool_call
            if not self._mcp_handler.has_tool_call(current_response):
                break
            
            # Распарси вызов
            tool_request = self._mcp_handler.parse_tool_call(current_response)
            if tool_request is None:
                break
            
            print(f"\n[Вызов инструмента: {tool_request.tool_name}]")
            
            # Выполни инструмент
            result = self._mcp_handler.call_tool(tool_request)
            
            # Форматируй результат
            formatted_result = self._mcp_handler.format_tool_result(
                tool_request.tool_name, result
            )
            
            # Отправь результат обратно в LLM
            current_response = self._llm_client.send_tool_result(
                tool_request.tool_name, 
                formatted_result
            )
        
        # Safety: strip any remaining tool_call tags if loop broke early
        if self._mcp_handler.has_tool_call(current_response):
            import re
            current_response = re.sub(r'<tool_call>.*?</tool_call>', '', current_response, flags=re.DOTALL).strip()
        
        return current_response
    
    def _search_knowledge_base(self, query: str) -> dict:
        """Обработчик для инструмента search_knowledge_base."""
        try:
            results = self._retriever.search(query, top_k=3)
            formatted = self._retriever.format_results_for_llm(results)
            return {"success": True, "results": formatted}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _recommend_tasks(self, priority: Optional[List[str]] = None,
                         status: Optional[List[str]] = None) -> dict:
        """
        Детерминированная рекомендация задач.

        Алгоритм:
        - вызвать MCP list_tasks с фильтрами
        - исключить статус done
        - отсортировать по priority, blocked_by, due_date
        - вернуть top_tasks, reasoning, notes
        """
        filters: Dict[str, Any] = {}
        if priority:
            filters["priority"] = list(priority)
        if status:
            filters["status"] = list(status)

        request = ToolCallRequest(tool_name="list_tasks", parameters=filters)
        result = self._mcp_handler.call_tool(request)
        if not result.success:
            return {"success": False, "error": result.error_message}

        tasks = self._extract_tasks(result.result)
        if not isinstance(tasks, list):
            return {"success": False, "error": "Непредвиденный формат результата list_tasks"}

        normalized = [self._normalize_task(task) for task in tasks if isinstance(task, dict)]

        # Фильтруем done и применяем явные фильтры, если list_tasks их не обработал
        filtered = []
        for task in normalized:
            if task["status"] == "done":
                continue
            if priority and task["priority"] and task["priority"] not in [p.lower() for p in priority]:
                continue
            if status and task["status"] and task["status"] not in [s.lower() for s in status]:
                continue
            filtered.append(task)

        ranked = sorted(filtered, key=self._task_sort_key)
        top_tasks = [self._present_task(task) for task in ranked[:3]]

        reasoning = []
        for task in ranked[:3]:
            reasoning.append(self._build_reasoning(task))

        notes = [
            "Сортировка: status (исключая done) → priority → blocked_by → due_date",
            "Приоритеты: critical > high > medium > low"
        ]

        return {
            "success": True,
            "top_tasks": top_tasks,
            "reasoning": reasoning,
            "notes": notes
        }

    def _extract_tasks(self, raw_result: Any) -> List[Dict[str, Any]]:
        """Пытается извлечь список задач из результата list_tasks."""
        if isinstance(raw_result, list):
            return raw_result

        if isinstance(raw_result, dict):
            for key in ("tasks", "items", "data", "result"):
                value = raw_result.get(key)
                if isinstance(value, list):
                    return value

            content = raw_result.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "text":
                        continue
                    text = item.get("text", "")
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict):
                        for key in ("tasks", "items", "data", "result"):
                            value = parsed.get(key)
                            if isinstance(value, list):
                                return value

        return []

    def _normalize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация полей задачи для сортировки."""
        task_id = task.get("id") or task.get("task_id") or task.get("key")
        title = task.get("title") or task.get("summary") or task.get("name")
        status = (task.get("status") or "").lower()
        priority = (task.get("priority") or "").lower()
        blocked_by = task.get("blocked_by") or task.get("blockedBy") or []
        due_date_raw = task.get("due_date") or task.get("dueDate") or task.get("deadline")

        blocked = False
        if isinstance(blocked_by, list):
            blocked = len(blocked_by) > 0
        elif isinstance(blocked_by, str):
            blocked = blocked_by.strip() != ""
        elif blocked_by:
            blocked = True

        due_date = None
        if isinstance(due_date_raw, str):
            try:
                due_date = datetime.fromisoformat(due_date_raw.replace("Z", "+00:00"))
            except ValueError:
                due_date = None

        return {
            "id": task_id,
            "title": title,
            "status": status,
            "priority": priority,
            "blocked": blocked,
            "blocked_by": blocked_by,
            "due_date": due_date,
            "due_date_raw": due_date_raw
        }

    def _task_sort_key(self, task: Dict[str, Any]) -> tuple:
        """Ключ сортировки для рекомендаций."""
        status_rank = {
            "in_progress": 0,
            "open": 1,
            "todo": 2,
            "blocked": 3,
            "on_hold": 4
        }
        priority_rank = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3
        }
        return (
            status_rank.get(task["status"], 5),
            priority_rank.get(task["priority"], 4),
            1 if task["blocked"] else 0,
            task["due_date"] or datetime.max,
            task["title"] or ""
        )

    def _present_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Упрощенный вид задачи для ответа инструмента."""
        return {
            "id": task["id"],
            "title": task["title"],
            "status": task["status"],
            "priority": task["priority"],
            "due_date": task["due_date_raw"],
            "blocked_by": task["blocked_by"]
        }

    def _build_reasoning(self, task: Dict[str, Any]) -> str:
        """Короткое объяснение позиции задачи в рейтинге."""
        parts = []
        if task["priority"]:
            parts.append(f"priority={task['priority']}")
        if task["status"]:
            parts.append(f"status={task['status']}")
        if task["blocked"]:
            parts.append("blocked_by есть")
        if task["due_date_raw"]:
            parts.append(f"due_date={task['due_date_raw']}")
        details = ", ".join(parts) if parts else "нет метаданных"
        title = task["title"] or task["id"] or "задача"
        return f"{title}: {details}"
    
    def clear_history(self) -> None:
        """
        Очистка истории диалога.
        
        Действия:
        - Очистить список сообщений
        - Сохранить системный промпт
        """
        self._llm_client.clear_history()
    
    def print_welcome(self) -> None:
        """
        Вывод приветственного сообщения.
        
        Действия:
        - Вывести ASCII баннер
        - Показать доступные команды
        - Показать версию приложения
        """
        print("""
╔════════════════════════════════════════════════╗
║         SUPPORT ASSISTANT v1.0                 ║
║     Ассистент службы поддержки                 ║
╚════════════════════════════════════════════════╝

Доступные команды:
  /index          - Индексировать документацию
  /clear          - Очистить историю диалога
  /status <id>    - Проверить статус тикета
  /help           - Показать справку
  /exit           - Выход

Начните диалог с приветствия!
    """)
    
    def print_help(self) -> None:
        """
        Вывод справки по командам.
        
        Действия:
        - Показать список всех доступных команд
        - Описать каждую команду
        """
        print("""
Справка по командам:

  /index
    Запускает индексацию всех документов в папке docs/
    Документы разбиваются на чанки и сохраняются с эмбедингами

  /clear
    Очищает историю текущего диалога
    Системный промпт сохраняется

  /status <ticket_id>
    Проверяет статус тикета по его ID
    Пример: /status TKT-12345

  /help
    Показывает эту справку

  /exit или /quit
    Завершает работу программы

Для общения с ассистентом просто введите ваш вопрос.
    """)


def main() -> None:
    """
    Точка входа в приложение.
    
    Действия:
    - Создать экземпляр SupportAssistant
    - Запустить консольный интерфейс
    - Обработать исключения верхнего уровня
    """
    try:
        assistant = SupportAssistant()
        assistant.start()
    except FileNotFoundError as e:
        print(f"Ошибка: не найден файл конфигурации - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
