"""
Главный модуль приложения Support Assistant.

Содержит точку входа и консольный интерфейс для взаимодействия
с пользователем.
"""

import os
import sys
from typing import Optional

import yaml

from llm_client import LLMClient
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
        
        # 5. Получение списка инструментов от MCP сервера
        tools = self._fetch_mcp_tools()
        
        # 6. LLM Client с динамическим системным промптом
        self._llm_client = LLMClient(
            api_key=self._api_config['perplexity']['api_key'],
            system_prompt=get_system_prompt(tools_override=tools)
        )
    
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
        
        return current_response
    
    def _search_knowledge_base(self, query: str) -> dict:
        """Обработчик для инструмента search_knowledge_base."""
        try:
            results = self._retriever.search(query, top_k=3)
            formatted = self._retriever.format_results_for_llm(results)
            return {"success": True, "results": formatted}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
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
