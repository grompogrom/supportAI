"""
Тесты для модуля поиска релевантных документов.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import sys
import os
import json
import tempfile

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.retriever import (
    DocumentRetriever,
    SearchResult,
    RetrieverError,
    IndexNotFoundError,
    IndexCorruptedError,
)


class TestSearchResult(unittest.TestCase):
    """Тесты для SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Проверка создания результата поиска."""
        result = SearchResult(
            chunk_id="chunk_001",
            text="Пример текста чанка",
            source_file="docs/example.md",
            similarity_score=0.95,
            position=0
        )
        
        self.assertEqual(result.chunk_id, "chunk_001")
        self.assertEqual(result.text, "Пример текста чанка")
        self.assertEqual(result.source_file, "docs/example.md")
        self.assertEqual(result.similarity_score, 0.95)
        self.assertEqual(result.position, 0)


class TestDocumentRetrieverInit(unittest.TestCase):
    """Тесты инициализации DocumentRetriever."""
    
    def test_init(self):
        """Проверка инициализации retriever'а."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        
        self.assertEqual(retriever._embeddings_path, "data/embeddings.json")
        self.assertEqual(retriever._embedding_generator, mock_generator)
        self.assertIsNone(retriever._index)
        self.assertIsNone(retriever._index_mtime)


class TestDocumentRetrieverCosineSimilarity(unittest.TestCase):
    """Тесты для вычисления косинусного сходства."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.mock_generator = Mock()
        self.retriever = DocumentRetriever("data/embeddings.json", self.mock_generator)
    
    def test_cosine_similarity_identical_vectors(self):
        """Проверка сходства идентичных векторов."""
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        similarity = self.retriever._compute_cosine_similarity(vec, vec)
        
        self.assertAlmostEqual(similarity, 1.0, places=6)
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Проверка сходства ортогональных векторов."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.retriever._compute_cosine_similarity(vec1, vec2)
        
        self.assertAlmostEqual(similarity, 0.0, places=6)
    
    def test_cosine_similarity_opposite_vectors(self):
        """Проверка сходства противоположных векторов."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = self.retriever._compute_cosine_similarity(vec1, vec2)
        
        self.assertAlmostEqual(similarity, -1.0, places=6)
    
    def test_cosine_similarity_zero_vector(self):
        """Проверка сходства с нулевым вектором."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = self.retriever._compute_cosine_similarity(vec1, vec2)
        
        self.assertEqual(similarity, 0.0)
    
    def test_cosine_similarity_known_value(self):
        """Проверка известного значения косинусного сходства."""
        # Векторы под углом 60 градусов: cos(60°) ≈ 0.5
        vec1 = [1.0, 0.0]
        vec2 = [0.5, 0.866025]  # cos(60°), sin(60°)
        similarity = self.retriever._compute_cosine_similarity(vec1, vec2)
        
        self.assertAlmostEqual(similarity, 0.5, places=4)


class TestDocumentRetrieverLoadIndex(unittest.TestCase):
    """Тесты для загрузки индекса."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.mock_generator = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
    
    def tearDown(self):
        """Очистка после тестов."""
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        os.rmdir(self.temp_dir)
    
    def test_load_index_file_not_found(self):
        """Проверка загрузки несуществующего индекса."""
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        
        result = retriever.load_index()
        
        self.assertFalse(result)
        self.assertIsNone(retriever._index)
    
    def test_load_index_success(self):
        """Проверка успешной загрузки индекса."""
        # Создаем тестовый индекс
        test_index = {
            "indexed_at": "2024-01-01T00:00:00",
            "config": {"chunk_size": 500},
            "chunks": [
                {
                    "id": "chunk_001",
                    "text": "Тестовый текст",
                    "source": "docs/test.md",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3]
                }
            ]
        }
        
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index, f)
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        result = retriever.load_index()
        
        self.assertTrue(result)
        self.assertIsNotNone(retriever._index)
        self.assertEqual(len(retriever._index["chunks"]), 1)
        self.assertIsNotNone(retriever._index_mtime)
    
    def test_load_index_empty_chunks(self):
        """Проверка загрузки индекса с пустыми чанками."""
        test_index = {
            "indexed_at": "2024-01-01T00:00:00",
            "chunks": []
        }
        
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index, f)
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        
        with self.assertRaises(IndexCorruptedError) as context:
            retriever.load_index()
        
        self.assertIn("пуст или поврежден", str(context.exception))
    
    def test_load_index_no_chunks_key(self):
        """Проверка загрузки индекса без ключа chunks."""
        test_index = {
            "indexed_at": "2024-01-01T00:00:00"
        }
        
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index, f)
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        
        with self.assertRaises(IndexCorruptedError):
            retriever.load_index()


class TestDocumentRetrieverIsIndexLoaded(unittest.TestCase):
    """Тесты для проверки загруженности индекса."""
    
    def test_is_index_loaded_false(self):
        """Проверка когда индекс не загружен."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        
        self.assertFalse(retriever.is_index_loaded())
    
    def test_is_index_loaded_true(self):
        """Проверка когда индекс загружен."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        retriever._index = {"chunks": []}
        
        self.assertTrue(retriever.is_index_loaded())


class TestDocumentRetrieverSearch(unittest.TestCase):
    """Тесты для поиска документов."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.mock_generator = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        
        # Создаем тестовый индекс с разными эмбедингами
        self.test_index = {
            "indexed_at": "2024-01-01T00:00:00",
            "config": {"chunk_size": 500},
            "chunks": [
                {
                    "id": "chunk_001",
                    "text": "Как сбросить пароль",
                    "source": "docs/passwords.md",
                    "position": 0,
                    "embedding": [1.0, 0.0, 0.0]  # Вектор в направлении X
                },
                {
                    "id": "chunk_002",
                    "text": "Установка программы",
                    "source": "docs/install.md",
                    "position": 0,
                    "embedding": [0.0, 1.0, 0.0]  # Вектор в направлении Y
                },
                {
                    "id": "chunk_003",
                    "text": "Восстановление пароля",
                    "source": "docs/passwords.md",
                    "position": 1,
                    "embedding": [0.9, 0.1, 0.0]  # Близко к X
                }
            ]
        }
        
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_index, f)
    
    def tearDown(self):
        """Очистка после тестов."""
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        os.rmdir(self.temp_dir)
    
    def test_search_returns_sorted_results(self):
        """Проверка сортировки результатов по релевантности."""
        # Мок генератора возвращает вектор близкий к chunk_001 и chunk_003
        self.mock_generator.generate.return_value = [1.0, 0.0, 0.0]
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        results = retriever.search("сбросить пароль", top_k=3)
        
        # Проверяем что результаты отсортированы по убыванию
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(results[0].similarity_score, results[1].similarity_score)
        self.assertGreaterEqual(results[1].similarity_score, results[2].similarity_score)
        
        # Первый результат должен быть chunk_001 (идеальное совпадение)
        self.assertEqual(results[0].chunk_id, "chunk_001")
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=4)
    
    def test_search_top_k(self):
        """Проверка ограничения количества результатов."""
        self.mock_generator.generate.return_value = [1.0, 0.0, 0.0]
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        results = retriever.search("тест", top_k=2)
        
        self.assertEqual(len(results), 2)
    
    def test_search_returns_search_result_objects(self):
        """Проверка типа возвращаемых объектов."""
        self.mock_generator.generate.return_value = [1.0, 0.0, 0.0]
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        results = retriever.search("тест", top_k=1)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.chunk_id, "chunk_001")
        self.assertEqual(result.text, "Как сбросить пароль")
        self.assertEqual(result.source_file, "docs/passwords.md")
        self.assertEqual(result.position, 0)
    
    def test_search_index_not_found(self):
        """Проверка поиска без индекса."""
        retriever = DocumentRetriever("/nonexistent/path.json", self.mock_generator)
        
        with self.assertRaises(IndexNotFoundError) as context:
            retriever.search("тест")
        
        self.assertIn("Индекс не найден", str(context.exception))


class TestDocumentRetrieverGetIndexStats(unittest.TestCase):
    """Тесты для получения статистики индекса."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.mock_generator = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        
        # Создаем тестовый индекс
        self.test_index = {
            "indexed_at": "2024-01-01T12:00:00",
            "config": {"chunk_size": 500, "chunk_overlap": 50},
            "chunks": [
                {"id": "1", "text": "t1", "source": "docs/a.md", "position": 0, "embedding": [0.1]},
                {"id": "2", "text": "t2", "source": "docs/a.md", "position": 1, "embedding": [0.2]},
                {"id": "3", "text": "t3", "source": "docs/b.md", "position": 0, "embedding": [0.3]},
            ]
        }
        
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_index, f)
    
    def tearDown(self):
        """Очистка после тестов."""
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        os.rmdir(self.temp_dir)
    
    def test_get_index_stats(self):
        """Проверка получения статистики."""
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        stats = retriever.get_index_stats()
        
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["indexed_at"], "2024-01-01T12:00:00")
        self.assertEqual(sorted(stats["source_files"]), ["docs/a.md", "docs/b.md"])
        self.assertEqual(stats["config"]["chunk_size"], 500)
    
    def test_get_index_stats_not_found(self):
        """Проверка статистики несуществующего индекса."""
        retriever = DocumentRetriever("/nonexistent/path.json", self.mock_generator)
        
        with self.assertRaises(IndexNotFoundError):
            retriever.get_index_stats()


class TestDocumentRetrieverFormatResults(unittest.TestCase):
    """Тесты для форматирования результатов."""
    
    def test_format_results_for_llm(self):
        """Проверка форматирования результатов для LLM."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        
        results = [
            SearchResult(
                chunk_id="chunk_001",
                text="Как сбросить пароль:\n1. Откройте настройки\n2. Нажмите 'Забыли пароль'",
                source_file="docs/passwords.md",
                similarity_score=0.95,
                position=0
            ),
            SearchResult(
                chunk_id="chunk_002",
                text="Восстановление пароля через email",
                source_file="docs/passwords.md",
                similarity_score=0.82,
                position=1
            ),
        ]
        
        formatted = retriever.format_results_for_llm(results)
        
        self.assertIn("Найденные документы:", formatted)
        self.assertIn("[1] Источник: docs/passwords.md", formatted)
        self.assertIn("релевантность: 0.95", formatted)
        self.assertIn("Как сбросить пароль:", formatted)
        self.assertIn("[2] Источник: docs/passwords.md", formatted)
        self.assertIn("релевантность: 0.82", formatted)
    
    def test_format_results_empty(self):
        """Проверка форматирования пустых результатов."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        
        formatted = retriever.format_results_for_llm([])
        
        self.assertEqual(formatted, "Найденные документы:")


class TestDocumentRetrieverReloadIndex(unittest.TestCase):
    """Тесты для автоматической перезагрузки индекса."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.mock_generator = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
    
    def tearDown(self):
        """Очистка после тестов."""
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        os.rmdir(self.temp_dir)
    
    def test_reload_index_if_needed_first_load(self):
        """Проверка первой загрузки индекса."""
        test_index = {
            "chunks": [{"id": "1", "text": "t", "source": "f", "position": 0, "embedding": [0.1]}]
        }
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index, f)
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        self.assertIsNone(retriever._index)
        
        retriever._reload_index_if_needed()
        
        self.assertIsNotNone(retriever._index)
    
    def test_reload_index_if_needed_file_changed(self):
        """Проверка перезагрузки при изменении файла."""
        import time
        
        # Создаем начальный индекс
        test_index_v1 = {
            "chunks": [{"id": "1", "text": "v1", "source": "f", "position": 0, "embedding": [0.1]}]
        }
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index_v1, f)
        
        retriever = DocumentRetriever(self.embeddings_path, self.mock_generator)
        retriever.load_index()
        
        self.assertEqual(retriever._index["chunks"][0]["text"], "v1")
        
        # Ждем немного чтобы mtime изменился
        time.sleep(0.1)
        
        # Обновляем индекс
        test_index_v2 = {
            "chunks": [{"id": "1", "text": "v2", "source": "f", "position": 0, "embedding": [0.1]}]
        }
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(test_index_v2, f)
        
        # Вызываем перезагрузку
        retriever._reload_index_if_needed()
        
        self.assertEqual(retriever._index["chunks"][0]["text"], "v2")


class TestRetrieverErrors(unittest.TestCase):
    """Тесты для классов ошибок."""
    
    def test_retriever_error_inheritance(self):
        """Проверка иерархии ошибок."""
        self.assertTrue(issubclass(IndexNotFoundError, RetrieverError))
        self.assertTrue(issubclass(IndexCorruptedError, RetrieverError))
    
    def test_index_not_found_error(self):
        """Проверка IndexNotFoundError."""
        error = IndexNotFoundError("Индекс не найден")
        self.assertIn("не найден", str(error))
    
    def test_index_corrupted_error(self):
        """Проверка IndexCorruptedError."""
        error = IndexCorruptedError("Индекс поврежден")
        self.assertIn("поврежден", str(error))


class TestComputeAllSimilarities(unittest.TestCase):
    """Тесты для вычисления сходства со всеми чанками."""
    
    def test_compute_all_similarities(self):
        """Проверка вычисления сходства со всеми чанками."""
        mock_generator = Mock()
        retriever = DocumentRetriever("data/embeddings.json", mock_generator)
        
        # Устанавливаем тестовый индекс напрямую
        retriever._index = {
            "chunks": [
                {"id": "1", "embedding": [1.0, 0.0, 0.0]},
                {"id": "2", "embedding": [0.0, 1.0, 0.0]},
                {"id": "3", "embedding": [0.5, 0.5, 0.0]},
            ]
        }
        
        query_embedding = [1.0, 0.0, 0.0]
        similarities = retriever._compute_all_similarities(query_embedding)
        
        self.assertEqual(len(similarities), 3)
        
        # Первый чанк должен иметь сходство 1.0
        self.assertEqual(similarities[0][0], 0)  # индекс
        self.assertAlmostEqual(similarities[0][1], 1.0, places=4)  # сходство
        
        # Второй чанк должен иметь сходство 0.0
        self.assertEqual(similarities[1][0], 1)
        self.assertAlmostEqual(similarities[1][1], 0.0, places=4)


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование модуля retriever")
    print("=" * 60)
    
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
