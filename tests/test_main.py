"""
Тесты для главного модуля main.py.

Включает unit-тесты и интеграционные тесты.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import json

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import (
    load_config,
    SupportAssistant,
    main,
)


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def temp_config_dir():
    """Создает временную директорию с тестовыми конфигами."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = os.path.join(tmpdir, 'config')
        os.makedirs(config_dir)
        
        # api_keys.yaml
        api_keys = {
            'perplexity': {'api_key': 'test-api-key'}
        }
        with open(os.path.join(config_dir, 'api_keys.yaml'), 'w') as f:
            import yaml
            yaml.dump(api_keys, f)
        
        # mcp_config.yaml
        mcp_config = {
            'servers': {
                'ticket_service': {
                    'host': 'localhost',
                    'port': 8000,
                    'endpoint': '/mcp'
                }
            }
        }
        with open(os.path.join(config_dir, 'mcp_config.yaml'), 'w') as f:
            yaml.dump(mcp_config, f)
        
        # local_llm_config.yaml
        llm_config = {
            'embedding_model': {
                'host': 'localhost',
                'port': 11434,
                'model_name': 'nomic-embed-text',
                'endpoint': '/api/embeddings'
            }
        }
        with open(os.path.join(config_dir, 'local_llm_config.yaml'), 'w') as f:
            yaml.dump(llm_config, f)
        
        # docs directory
        docs_dir = os.path.join(tmpdir, 'docs')
        os.makedirs(docs_dir)
        
        # data directory
        data_dir = os.path.join(tmpdir, 'data')
        os.makedirs(data_dir)
        
        yield tmpdir


@pytest.fixture
def mock_llm_client():
    """Мок LLM клиента."""
    with patch('main.LLMClient') as mock:
        instance = Mock()
        instance.send_message.return_value = "Привет! Чем могу помочь?"
        instance.send_tool_result.return_value = "Инструмент выполнен успешно."
        instance.clear_history.return_value = None
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_embedding_generator():
    """Мок генератора эмбедингов."""
    with patch('main.EmbeddingGenerator') as mock:
        instance = Mock()
        instance.generate.return_value = [0.1] * 768
        instance.generate_batch.return_value = [[0.1] * 768]
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_document_retriever():
    """Мок retriever'а."""
    with patch('main.DocumentRetriever') as mock:
        instance = Mock()
        instance.search.return_value = []
        instance.format_results_for_llm.return_value = "Документы не найдены"
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_mcp_handler():
    """Мок MCP обработчика."""
    with patch('main.MCPHandler') as mock:
        instance = Mock()
        instance.has_tool_call.return_value = False
        instance.parse_tool_call.return_value = None
        instance.call_tool.return_value = Mock(success=True, result={"status": "ok"})
        instance.format_tool_result.return_value = "<tool_result>OK</tool_result>"
        instance.register_local_tool.return_value = None
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_document_indexer():
    """Мок индексатора."""
    with patch('main.DocumentIndexer') as mock:
        instance = Mock()
        index_result = Mock()
        index_result.total_files = 2
        index_result.total_chunks = 10
        index_result.errors = []
        instance.index_all.return_value = index_result
        mock.return_value = instance
        yield mock


@pytest.fixture
def assistant_with_mocks(
    temp_config_dir,
    mock_llm_client,
    mock_embedding_generator,
    mock_document_retriever,
    mock_mcp_handler,
    mock_document_indexer
):
    """SupportAssistant с замоканными зависимостями."""
    # Патчим os.path для использования временной директории
    original_abspath = os.path.abspath
    
    def mock_abspath(path):
        if path.endswith('main.py'):
            return os.path.join(temp_config_dir, 'src', 'main.py')
        return original_abspath(path)
    
    with patch('main.os.path.abspath', side_effect=mock_abspath):
        with patch('main.os.path.dirname') as mock_dirname:
            # Первый вызов - директория с main.py (src/)
            # Второй вызов - базовая директория проекта
            mock_dirname.side_effect = lambda p: (
                os.path.join(temp_config_dir, 'src') if 'main.py' in p 
                else temp_config_dir
            )
            assistant = SupportAssistant()
    
    return assistant


# ============================================================================
# Тесты load_config
# ============================================================================

class TestLoadConfig:
    """Тесты функции load_config."""
    
    def test_load_valid_yaml(self, temp_config_dir):
        """Загрузка валидного YAML файла."""
        config_path = os.path.join(temp_config_dir, 'config', 'api_keys.yaml')
        config = load_config(config_path)
        
        assert config is not None
        assert 'perplexity' in config
        assert config['perplexity']['api_key'] == 'test-api-key'
    
    def test_load_nonexistent_file(self):
        """Попытка загрузить несуществующий файл."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')
    
    def test_load_empty_yaml(self, temp_config_dir):
        """Загрузка пустого YAML файла."""
        empty_path = os.path.join(temp_config_dir, 'config', 'empty.yaml')
        with open(empty_path, 'w') as f:
            f.write('')
        
        config = load_config(empty_path)
        assert config is None
    
    def test_load_complex_yaml(self, temp_config_dir):
        """Загрузка сложной YAML структуры."""
        import yaml
        complex_config = {
            'servers': {
                'server1': {'host': 'localhost', 'port': 8000},
                'server2': {'host': 'localhost', 'port': 8001}
            },
            'features': ['feature1', 'feature2'],
            'nested': {
                'level1': {
                    'level2': 'value'
                }
            }
        }
        
        complex_path = os.path.join(temp_config_dir, 'config', 'complex.yaml')
        with open(complex_path, 'w') as f:
            yaml.dump(complex_config, f)
        
        config = load_config(complex_path)
        
        assert config['servers']['server1']['port'] == 8000
        assert 'feature1' in config['features']
        assert config['nested']['level1']['level2'] == 'value'


# ============================================================================
# Тесты SupportAssistant инициализации
# ============================================================================

class TestSupportAssistantInit:
    """Тесты инициализации SupportAssistant."""
    
    def test_initialization_success(self, assistant_with_mocks):
        """Успешная инициализация ассистента."""
        assistant = assistant_with_mocks
        
        assert assistant is not None
        assert assistant._llm_client is not None
        assert assistant._mcp_handler is not None
        assert assistant._indexer is not None
        assert assistant._retriever is not None
        assert assistant._embedding_generator is not None
    
    def test_configs_loaded(self, assistant_with_mocks):
        """Конфигурации загружены корректно."""
        assistant = assistant_with_mocks
        
        assert assistant._api_config is not None
        assert assistant._mcp_config is not None
        assert assistant._llm_config is not None
    
    def test_local_tool_registered(self, assistant_with_mocks, mock_mcp_handler):
        """Локальный инструмент search_knowledge_base зарегистрирован."""
        # Проверяем что register_local_tool был вызван
        mock_mcp_handler.return_value.register_local_tool.assert_called_once()
        
        call_args = mock_mcp_handler.return_value.register_local_tool.call_args
        assert call_args[1]['name'] == 'search_knowledge_base'


# ============================================================================
# Тесты обработки команд
# ============================================================================

class TestHandleCommand:
    """Тесты обработки команд."""
    
    def test_command_help(self, assistant_with_mocks, capsys):
        """Команда /help выводит справку."""
        assistant = assistant_with_mocks
        result = assistant.handle_command('/help')
        
        # /help возвращает None и выводит справку в консоль
        assert result is None
        
        captured = capsys.readouterr()
        assert '/index' in captured.out
        assert '/clear' in captured.out
        assert '/status' in captured.out
    
    def test_command_clear(self, assistant_with_mocks):
        """Команда /clear очищает историю."""
        assistant = assistant_with_mocks
        result = assistant.handle_command('/clear')
        
        assert result == "История диалога очищена."
        assistant._llm_client.clear_history.assert_called_once()
    
    def test_command_index(self, assistant_with_mocks, capsys):
        """Команда /index запускает индексацию."""
        assistant = assistant_with_mocks
        result = assistant.handle_command('/index')
        
        assert 'Индексация завершена' in result
        assert 'Файлов: 2' in result
        assert 'Чанков: 10' in result
    
    def test_command_status_with_arg(self, assistant_with_mocks):
        """Команда /status с аргументом."""
        assistant = assistant_with_mocks
        
        # Настраиваем мок
        assistant._mcp_handler.call_tool.return_value = Mock(
            success=True,
            result="открыт"
        )
        
        result = assistant.handle_command('/status TKT-123')
        
        assert 'TKT-123' in result
    
    def test_command_status_without_arg(self, assistant_with_mocks):
        """Команда /status без аргумента."""
        assistant = assistant_with_mocks
        result = assistant.handle_command('/status')
        
        # Без аргумента команда не распознается как /status
        assert 'Неизвестная команда' in result
    
    def test_command_exit(self, assistant_with_mocks):
        """Команда /exit завершает программу."""
        assistant = assistant_with_mocks
        
        with pytest.raises(SystemExit) as exc_info:
            assistant.handle_command('/exit')
        
        assert exc_info.value.code == 0
    
    def test_command_quit(self, assistant_with_mocks):
        """Команда /quit завершает программу."""
        assistant = assistant_with_mocks
        
        with pytest.raises(SystemExit):
            assistant.handle_command('/quit')
    
    def test_unknown_command(self, assistant_with_mocks):
        """Неизвестная команда возвращает ошибку."""
        assistant = assistant_with_mocks
        result = assistant.handle_command('/unknown')
        
        assert 'Неизвестная команда' in result
        assert '/unknown' in result
        assert '/help' in result
    
    def test_command_case_insensitive(self, assistant_with_mocks, capsys):
        """Команды нечувствительны к регистру."""
        assistant = assistant_with_mocks
        
        result = assistant.handle_command('/HELP')
        assert result is None  # /help возвращает None
        
        result = assistant.handle_command('/CLEAR')
        assert result == "История диалога очищена."


# ============================================================================
# Тесты process_input
# ============================================================================

class TestProcessInput:
    """Тесты обработки пользовательского ввода."""
    
    def test_process_command(self, assistant_with_mocks):
        """Ввод начинающийся с / обрабатывается как команда."""
        assistant = assistant_with_mocks
        result = assistant.process_input('/clear')
        
        assert result == "История диалога очищена."
    
    def test_process_message(self, assistant_with_mocks):
        """Обычный текст отправляется в LLM."""
        assistant = assistant_with_mocks
        assistant._llm_client.send_message.return_value = "Привет!"
        assistant._mcp_handler.has_tool_call.return_value = False
        
        result = assistant.process_input('Привет')
        
        assistant._llm_client.send_message.assert_called_with('Привет')
        assert result == "Привет!"


# ============================================================================
# Тесты send_to_llm
# ============================================================================

class TestSendToLLM:
    """Тесты отправки сообщений в LLM."""
    
    def test_send_simple_message(self, assistant_with_mocks):
        """Отправка простого сообщения."""
        assistant = assistant_with_mocks
        assistant._llm_client.send_message.return_value = "Ответ LLM"
        assistant._mcp_handler.has_tool_call.return_value = False
        
        result = assistant.send_to_llm("Привет")
        
        assert result == "Ответ LLM"
    
    def test_send_message_with_tool_call(self, assistant_with_mocks):
        """Сообщение с tool_call обрабатывается."""
        assistant = assistant_with_mocks
        
        # Первый ответ содержит tool_call
        assistant._llm_client.send_message.return_value = "<tool_call>...</tool_call>"
        assistant._mcp_handler.has_tool_call.side_effect = [True, False]
        
        tool_request = Mock()
        tool_request.tool_name = "search_knowledge_base"
        assistant._mcp_handler.parse_tool_call.return_value = tool_request
        
        assistant._mcp_handler.call_tool.return_value = Mock(success=True, result="data")
        assistant._mcp_handler.format_tool_result.return_value = "<tool_result>data</tool_result>"
        assistant._llm_client.send_tool_result.return_value = "Финальный ответ"
        
        result = assistant.send_to_llm("Найди информацию")
        
        assert result == "Финальный ответ"


# ============================================================================
# Тесты process_tool_calls
# ============================================================================

class TestProcessToolCalls:
    """Тесты обработки tool_call."""
    
    def test_no_tool_call(self, assistant_with_mocks):
        """Ответ без tool_call возвращается как есть."""
        assistant = assistant_with_mocks
        assistant._mcp_handler.has_tool_call.return_value = False
        
        result = assistant.process_tool_calls("Обычный ответ")
        
        assert result == "Обычный ответ"
    
    def test_single_tool_call(self, assistant_with_mocks, capsys):
        """Обработка одного tool_call."""
        assistant = assistant_with_mocks
        
        # Первый вызов - есть tool_call, второй - нет
        assistant._mcp_handler.has_tool_call.side_effect = [True, False]
        
        tool_request = Mock()
        tool_request.tool_name = "test_tool"
        assistant._mcp_handler.parse_tool_call.return_value = tool_request
        
        tool_result = Mock(success=True, result="результат")
        assistant._mcp_handler.call_tool.return_value = tool_result
        assistant._mcp_handler.format_tool_result.return_value = "formatted"
        assistant._llm_client.send_tool_result.return_value = "Финальный ответ"
        
        result = assistant.process_tool_calls("<tool_call>...</tool_call>")
        
        assert result == "Финальный ответ"
        
        captured = capsys.readouterr()
        assert '[Вызов инструмента: test_tool]' in captured.out
    
    def test_max_iterations_limit(self, assistant_with_mocks):
        """Защита от бесконечного цикла tool_call."""
        assistant = assistant_with_mocks
        
        # has_tool_call всегда возвращает True
        assistant._mcp_handler.has_tool_call.return_value = True
        
        tool_request = Mock()
        tool_request.tool_name = "infinite_tool"
        assistant._mcp_handler.parse_tool_call.return_value = tool_request
        
        assistant._mcp_handler.call_tool.return_value = Mock(success=True, result="ok")
        assistant._mcp_handler.format_tool_result.return_value = "formatted"
        # Каждый send_tool_result возвращает ответ с tool_call
        assistant._llm_client.send_tool_result.return_value = "<tool_call>again</tool_call>"
        
        result = assistant.process_tool_calls("start")
        
        # Должно быть максимум 5 итераций
        assert assistant._llm_client.send_tool_result.call_count == 5
    
    def test_tool_call_parse_failure(self, assistant_with_mocks):
        """Если parse_tool_call возвращает None, цикл прекращается."""
        assistant = assistant_with_mocks
        
        assistant._mcp_handler.has_tool_call.return_value = True
        assistant._mcp_handler.parse_tool_call.return_value = None
        
        result = assistant.process_tool_calls("bad tool call")
        
        assert result == "bad tool call"


# ============================================================================
# Тесты _search_knowledge_base
# ============================================================================

class TestSearchKnowledgeBase:
    """Тесты локального инструмента поиска."""
    
    def test_search_success(self, assistant_with_mocks):
        """Успешный поиск в базе знаний."""
        assistant = assistant_with_mocks
        
        # Настраиваем мок retriever
        mock_results = [Mock(text="Найденный документ")]
        assistant._retriever.search.return_value = mock_results
        assistant._retriever.format_results_for_llm.return_value = "Форматированный результат"
        
        result = assistant._search_knowledge_base("тестовый запрос")
        
        assert result['success'] is True
        assert result['results'] == "Форматированный результат"
        assistant._retriever.search.assert_called_with("тестовый запрос", top_k=3)
    
    def test_search_error(self, assistant_with_mocks):
        """Ошибка при поиске."""
        assistant = assistant_with_mocks
        
        assistant._retriever.search.side_effect = Exception("Индекс не найден")
        
        result = assistant._search_knowledge_base("запрос")
        
        assert result['success'] is False
        assert 'Индекс не найден' in result['error']


# ============================================================================
# Тесты clear_history
# ============================================================================

class TestClearHistory:
    """Тесты очистки истории."""
    
    def test_clear_history(self, assistant_with_mocks):
        """Очистка истории вызывает метод LLM клиента."""
        assistant = assistant_with_mocks
        
        assistant.clear_history()
        
        assistant._llm_client.clear_history.assert_called_once()


# ============================================================================
# Тесты print_welcome и print_help
# ============================================================================

class TestPrintMethods:
    """Тесты методов вывода."""
    
    def test_print_welcome(self, assistant_with_mocks, capsys):
        """print_welcome выводит баннер."""
        assistant = assistant_with_mocks
        
        assistant.print_welcome()
        
        captured = capsys.readouterr()
        assert 'SUPPORT ASSISTANT' in captured.out
        assert '/index' in captured.out
        assert '/clear' in captured.out
        assert '/help' in captured.out
        assert '/exit' in captured.out
    
    def test_print_help(self, assistant_with_mocks, capsys):
        """print_help выводит справку."""
        assistant = assistant_with_mocks
        
        assistant.print_help()
        
        captured = capsys.readouterr()
        assert 'Справка по командам' in captured.out
        assert '/index' in captured.out
        assert '/clear' in captured.out
        assert '/status' in captured.out
        assert '/help' in captured.out
        assert '/exit' in captured.out
        assert '/quit' in captured.out


# ============================================================================
# Тесты main()
# ============================================================================

class TestMain:
    """Тесты точки входа."""
    
    def test_main_file_not_found(self, capsys):
        """main() обрабатывает FileNotFoundError."""
        with patch('main.SupportAssistant') as mock_assistant:
            mock_assistant.side_effect = FileNotFoundError("config.yaml")
            
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
            
            captured = capsys.readouterr()
            assert 'не найден файл конфигурации' in captured.out
    
    def test_main_generic_error(self, capsys):
        """main() обрабатывает общие исключения."""
        with patch('main.SupportAssistant') as mock_assistant:
            mock_assistant.side_effect = Exception("Unexpected error")
            
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
            
            captured = capsys.readouterr()
            assert 'Критическая ошибка' in captured.out


# ============================================================================
# Тесты _do_index
# ============================================================================

class TestDoIndex:
    """Тесты метода _do_index."""
    
    def test_do_index_success(self, assistant_with_mocks, capsys):
        """Успешная индексация."""
        assistant = assistant_with_mocks
        
        result = assistant._do_index()
        
        assert 'Индексация завершена' in result
        assert 'Файлов: 2' in result
        assert 'Чанков: 10' in result
        assert 'Ошибок: 0' in result
        
        captured = capsys.readouterr()
        assert 'Начинаю индексацию' in captured.out
    
    def test_do_index_with_errors(self, assistant_with_mocks):
        """Индексация с ошибками."""
        assistant = assistant_with_mocks
        
        # Настраиваем мок с ошибками
        index_result = Mock()
        index_result.total_files = 3
        index_result.total_chunks = 5
        index_result.errors = ["file1.txt: ошибка", "file2.txt: ошибка"]
        assistant._indexer.index_all.return_value = index_result
        
        result = assistant._do_index()
        
        assert 'Ошибок: 2' in result
    
    def test_do_index_exception(self, assistant_with_mocks):
        """Исключение при индексации."""
        assistant = assistant_with_mocks
        
        assistant._indexer.index_all.side_effect = Exception("Ollama не доступен")
        
        result = assistant._do_index()
        
        assert 'Ошибка индексации' in result
        assert 'Ollama не доступен' in result


# ============================================================================
# Тесты _check_ticket_status
# ============================================================================

class TestCheckTicketStatus:
    """Тесты метода _check_ticket_status."""
    
    def test_check_status_success(self, assistant_with_mocks):
        """Успешная проверка статуса."""
        assistant = assistant_with_mocks
        
        assistant._mcp_handler.call_tool.return_value = Mock(
            success=True,
            result="открыт"
        )
        
        result = assistant._check_ticket_status("TKT-123")
        
        assert 'TKT-123' in result
        assert 'открыт' in result
    
    def test_check_status_error(self, assistant_with_mocks):
        """Ошибка проверки статуса."""
        assistant = assistant_with_mocks
        
        assistant._mcp_handler.call_tool.return_value = Mock(
            success=False,
            error_message="Тикет не найден"
        )
        
        result = assistant._check_ticket_status("TKT-999")
        
        assert 'Ошибка' in result
        assert 'Тикет не найден' in result


# ============================================================================
# Интеграционные тесты с реальными конфигами
# ============================================================================

class TestIntegration:
    """Интеграционные тесты с реальными файлами конфигурации."""
    
    @pytest.fixture
    def project_root(self):
        """Путь к корню проекта."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_load_real_api_keys_config(self, project_root):
        """Загрузка реального api_keys.yaml."""
        config_path = os.path.join(project_root, 'config', 'api_keys.yaml')
        
        if not os.path.exists(config_path):
            pytest.skip("Файл api_keys.yaml не существует")
        
        config = load_config(config_path)
        
        assert config is not None
        assert 'perplexity' in config
        assert 'api_key' in config['perplexity']
    
    def test_load_real_mcp_config(self, project_root):
        """Загрузка реального mcp_config.yaml."""
        config_path = os.path.join(project_root, 'config', 'mcp_config.yaml')
        
        if not os.path.exists(config_path):
            pytest.skip("Файл mcp_config.yaml не существует")
        
        config = load_config(config_path)
        
        assert config is not None
        assert 'servers' in config
    
    def test_load_real_llm_config(self, project_root):
        """Загрузка реального local_llm_config.yaml."""
        config_path = os.path.join(project_root, 'config', 'local_llm_config.yaml')
        
        if not os.path.exists(config_path):
            pytest.skip("Файл local_llm_config.yaml не существует")
        
        config = load_config(config_path)
        
        assert config is not None
        assert 'embedding_model' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
