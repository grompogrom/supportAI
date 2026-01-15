"""
Тесты для модуля индексации документов.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import json
import sys

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.indexer import (
    DocumentIndexer,
    DocumentChunk,
    IndexingResult,
    IndexerError,
)


class TestDocumentChunk(unittest.TestCase):
    """Тесты для DocumentChunk dataclass."""
    
    def test_chunk_creation(self):
        """Проверка создания чанка."""
        chunk = DocumentChunk(
            chunk_id="test_chunk_0001",
            text="Sample text content",
            source_file="docs/test.md",
            position=0
        )
        self.assertEqual(chunk.chunk_id, "test_chunk_0001")
        self.assertEqual(chunk.text, "Sample text content")
        self.assertEqual(chunk.source_file, "docs/test.md")
        self.assertEqual(chunk.position, 0)


class TestIndexingResult(unittest.TestCase):
    """Тесты для IndexingResult dataclass."""
    
    def test_result_creation(self):
        """Проверка создания результата индексации."""
        result = IndexingResult(
            total_files=5,
            total_chunks=20,
            indexed_files=["file1.md", "file2.md"],
            errors=["file3.md: error"]
        )
        self.assertEqual(result.total_files, 5)
        self.assertEqual(result.total_chunks, 20)
        self.assertEqual(len(result.indexed_files), 2)
        self.assertEqual(len(result.errors), 1)


class TestDocumentIndexerInit(unittest.TestCase):
    """Тесты инициализации DocumentIndexer."""
    
    def setUp(self):
        """Создание временной директории для тестов."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        self.embeddings_path = os.path.join(self.temp_dir, "data", "embeddings.json")
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_docs_dir(self):
        """Проверка создания директории docs при инициализации."""
        self.assertFalse(os.path.exists(self.docs_dir))
        
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        
        self.assertTrue(os.path.exists(self.docs_dir))
    
    def test_init_stores_parameters(self):
        """Проверка сохранения параметров."""
        os.makedirs(self.docs_dir)
        
        indexer = DocumentIndexer(
            self.docs_dir,
            self.embeddings_path,
            chunk_size=100,
            overlap=20
        )
        
        self.assertEqual(indexer._docs_dir, self.docs_dir)
        self.assertEqual(indexer._embeddings_path, self.embeddings_path)
        self.assertEqual(indexer._chunk_size, 100)
        self.assertEqual(indexer._overlap, 20)
    
    def test_init_default_values(self):
        """Проверка значений по умолчанию."""
        os.makedirs(self.docs_dir)
        
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        
        self.assertEqual(indexer._chunk_size, 500)
        self.assertEqual(indexer._overlap, 50)


class TestScanDocuments(unittest.TestCase):
    """Тесты для scan_documents."""
    
    def setUp(self):
        """Создание временной директории с тестовыми файлами."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        
        # Создаём тестовые файлы
        self._create_file("test1.md", "# Test 1")
        self._create_file("test2.txt", "Test 2 content")
        self._create_file("test3.MD", "# Test 3 uppercase ext")
        self._create_file("ignored.py", "# Not a doc")
        self._create_file("ignored.json", "{}")
        
        # Создаём поддиректорию с файлами
        subdir = os.path.join(self.docs_dir, "subdir")
        os.makedirs(subdir)
        self._create_file("subdir/nested.md", "# Nested doc")
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def _create_file(self, relative_path, content):
        """Создание файла в docs директории."""
        full_path = os.path.join(self.docs_dir, relative_path)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def test_scan_finds_md_files(self):
        """Проверка поиска .md файлов."""
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        md_files = [f for f in files if f.endswith('.md') or f.endswith('.MD')]
        self.assertEqual(len(md_files), 3)  # test1.md, test3.MD, nested.md
    
    def test_scan_finds_txt_files(self):
        """Проверка поиска .txt файлов."""
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        txt_files = [f for f in files if f.endswith('.txt')]
        self.assertEqual(len(txt_files), 1)
    
    def test_scan_ignores_unsupported_extensions(self):
        """Проверка игнорирования неподдерживаемых расширений."""
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        py_files = [f for f in files if f.endswith('.py')]
        json_files = [f for f in files if f.endswith('.json')]
        
        self.assertEqual(len(py_files), 0)
        self.assertEqual(len(json_files), 0)
    
    def test_scan_recursive(self):
        """Проверка рекурсивного сканирования."""
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        nested_files = [f for f in files if 'subdir' in f]
        self.assertEqual(len(nested_files), 1)
    
    def test_scan_returns_sorted_list(self):
        """Проверка сортировки результатов."""
        indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        self.assertEqual(files, sorted(files))
    
    def test_scan_empty_directory(self):
        """Проверка сканирования пустой директории."""
        empty_dir = os.path.join(self.temp_dir, "empty_docs")
        os.makedirs(empty_dir)
        
        indexer = DocumentIndexer(empty_dir, self.embeddings_path)
        files = indexer.scan_documents()
        
        self.assertEqual(files, [])


class TestReadDocument(unittest.TestCase):
    """Тесты для read_document."""
    
    def setUp(self):
        """Создание временной директории."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        self.indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_read_utf8_file(self):
        """Проверка чтения UTF-8 файла."""
        file_path = os.path.join(self.docs_dir, "test.md")
        content = "# Заголовок\n\nТекст на русском языке."
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.indexer.read_document(file_path)
        
        self.assertEqual(result, content)
    
    def test_read_strips_whitespace(self):
        """Проверка удаления лишних пробелов."""
        file_path = os.path.join(self.docs_dir, "test.md")
        content = "  \n\nContent\n\n  "
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.indexer.read_document(file_path)
        
        self.assertEqual(result, "Content")
    
    def test_read_latin1_fallback(self):
        """Проверка fallback на latin-1 кодировку."""
        file_path = os.path.join(self.docs_dir, "test.txt")
        
        # Записываем файл в latin-1
        with open(file_path, 'wb') as f:
            f.write(b"Caf\xe9 content")  # latin-1 encoded "Café"
        
        result = self.indexer.read_document(file_path)
        
        self.assertIn("Caf", result)
    
    def test_read_nonexistent_file(self):
        """Проверка ошибки при чтении несуществующего файла."""
        file_path = os.path.join(self.docs_dir, "nonexistent.md")
        
        with self.assertRaises(FileNotFoundError):
            self.indexer.read_document(file_path)


class TestSplitIntoChunks(unittest.TestCase):
    """Тесты для split_into_chunks."""
    
    def setUp(self):
        """Создание временной директории."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_chunking_with_overlap_spec_example(self):
        """Проверка алгоритма чанкинга из спецификации."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=10, overlap=3
        )
        
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = list(indexer.split_into_chunks(text, "test.txt"))
        
        self.assertEqual(len(chunks), 4)
        
        # Chunk 1: ABCDEFGHIJ (position 0)
        self.assertEqual(chunks[0].text, "ABCDEFGHIJ")
        self.assertEqual(chunks[0].position, 0)
        
        # Chunk 2: HIJKLMNOPQ (position 7)
        self.assertEqual(chunks[1].text, "HIJKLMNOPQ")
        self.assertEqual(chunks[1].position, 7)
        
        # Chunk 3: OPQRSTUVWX (position 14)
        self.assertEqual(chunks[2].text, "OPQRSTUVWX")
        self.assertEqual(chunks[2].position, 14)
        
        # Chunk 4: VWXYZ (position 21, shorter)
        self.assertEqual(chunks[3].text, "VWXYZ")
        self.assertEqual(chunks[3].position, 21)
    
    def test_chunk_ids_are_unique(self):
        """Проверка уникальности chunk_id."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=10, overlap=3
        )
        
        text = "A" * 100
        chunks = list(indexer.split_into_chunks(text, "test.txt"))
        
        chunk_ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
    
    def test_chunk_id_format(self):
        """Проверка формата chunk_id."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=10, overlap=3
        )
        
        chunks = list(indexer.split_into_chunks("ABCDEFGHIJ", "example.md"))
        
        self.assertEqual(chunks[0].chunk_id, "example_chunk_0000")
    
    def test_single_chunk_for_short_text(self):
        """Проверка одного чанка для короткого текста."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=100, overlap=10
        )
        
        text = "Short text"
        chunks = list(indexer.split_into_chunks(text, "test.txt"))
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short text")
    
    def test_empty_text_no_chunks(self):
        """Проверка пустого текста."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=10, overlap=3
        )
        
        chunks = list(indexer.split_into_chunks("", "test.txt"))
        
        self.assertEqual(len(chunks), 0)
    
    def test_source_file_in_chunk(self):
        """Проверка сохранения source_file в чанке."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=10, overlap=3
        )
        
        source = "docs/subdir/myfile.md"
        chunks = list(indexer.split_into_chunks("ABCDEFGHIJ", source))
        
        self.assertEqual(chunks[0].source_file, source)


class TestGenerateChunkId(unittest.TestCase):
    """Тесты для _generate_chunk_id."""
    
    def setUp(self):
        """Создание временной директории."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        self.indexer = DocumentIndexer(self.docs_dir, self.embeddings_path)
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_chunk_id_format(self):
        """Проверка формата ID."""
        chunk_id = self.indexer._generate_chunk_id("docs/example.md", 0)
        self.assertEqual(chunk_id, "example_chunk_0000")
    
    def test_generate_chunk_id_with_position(self):
        """Проверка ID с разными позициями."""
        self.assertEqual(
            self.indexer._generate_chunk_id("test.txt", 0),
            "test_chunk_0000"
        )
        self.assertEqual(
            self.indexer._generate_chunk_id("test.txt", 1),
            "test_chunk_0001"
        )
        self.assertEqual(
            self.indexer._generate_chunk_id("test.txt", 99),
            "test_chunk_0099"
        )
        self.assertEqual(
            self.indexer._generate_chunk_id("test.txt", 9999),
            "test_chunk_9999"
        )
    
    def test_generate_chunk_id_with_path(self):
        """Проверка ID с полным путём к файлу."""
        chunk_id = self.indexer._generate_chunk_id(
            "/home/user/docs/subdir/myfile.md", 5
        )
        self.assertEqual(chunk_id, "myfile_chunk_0005")


class TestSaveAndLoadIndex(unittest.TestCase):
    """Тесты для save_index и load_index."""
    
    def setUp(self):
        """Создание временной директории."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "data", "embeddings.json")
        self.indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=100, overlap=10
        )
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_creates_directory(self):
        """Проверка создания директории при сохранении."""
        chunks = [
            DocumentChunk("chunk_0000", "Text 1", "file1.md", 0),
        ]
        embeddings = [[0.1, 0.2, 0.3]]
        
        self.assertFalse(os.path.exists(os.path.dirname(self.embeddings_path)))
        
        self.indexer.save_index(chunks, embeddings)
        
        self.assertTrue(os.path.exists(self.embeddings_path))
    
    def test_save_and_load_roundtrip(self):
        """Проверка сохранения и загрузки индекса."""
        chunks = [
            DocumentChunk("chunk_0000", "Text 1", "file1.md", 0),
            DocumentChunk("chunk_0001", "Text 2", "file2.md", 100),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        self.indexer.save_index(chunks, embeddings)
        loaded = self.indexer.load_index()
        
        self.assertIsNotNone(loaded)
        self.assertIn("indexed_at", loaded)
        self.assertIn("config", loaded)
        self.assertIn("chunks", loaded)
        
        self.assertEqual(loaded["config"]["chunk_size"], 100)
        self.assertEqual(loaded["config"]["overlap"], 10)
        
        self.assertEqual(len(loaded["chunks"]), 2)
        self.assertEqual(loaded["chunks"][0]["id"], "chunk_0000")
        self.assertEqual(loaded["chunks"][0]["text"], "Text 1")
        self.assertEqual(loaded["chunks"][0]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(loaded["chunks"][0]["source"], "file1.md")
        self.assertEqual(loaded["chunks"][0]["position"], 0)
    
    def test_load_nonexistent_returns_none(self):
        """Проверка загрузки несуществующего индекса."""
        result = self.indexer.load_index()
        self.assertIsNone(result)
    
    def test_is_index_exists_false(self):
        """Проверка is_index_exists для несуществующего файла."""
        self.assertFalse(self.indexer.is_index_exists())
    
    def test_is_index_exists_true(self):
        """Проверка is_index_exists для существующего файла."""
        chunks = [DocumentChunk("chunk_0000", "Text", "file.md", 0)]
        embeddings = [[0.1, 0.2]]
        
        self.indexer.save_index(chunks, embeddings)
        
        self.assertTrue(self.indexer.is_index_exists())
    
    def test_save_unicode_content(self):
        """Проверка сохранения Unicode контента."""
        chunks = [
            DocumentChunk("chunk_0000", "Текст на русском 🎉", "file.md", 0),
        ]
        embeddings = [[0.1, 0.2]]
        
        self.indexer.save_index(chunks, embeddings)
        loaded = self.indexer.load_index()
        
        self.assertEqual(loaded["chunks"][0]["text"], "Текст на русском 🎉")


class TestIndexAll(unittest.TestCase):
    """Тесты для index_all."""
    
    def setUp(self):
        """Создание временной директории с тестовыми файлами."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
        
        # Создаём тестовые документы
        with open(os.path.join(self.docs_dir, "doc1.md"), 'w') as f:
            f.write("First document content")
        with open(os.path.join(self.docs_dir, "doc2.txt"), 'w') as f:
            f.write("Second document content")
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_index_all_calls_embedding_generator(self):
        """Проверка вызова генератора эмбедингов."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=50, overlap=5
        )
        
        # Мок генератора эмбедингов
        mock_generator = Mock()
        mock_generator.generate.return_value = [0.1, 0.2, 0.3]
        
        result = indexer.index_all(mock_generator)
        
        # Должен быть вызван для каждого чанка
        self.assertGreater(mock_generator.generate.call_count, 0)
    
    def test_index_all_returns_correct_result(self):
        """Проверка возврата корректного результата."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=50, overlap=5
        )
        
        mock_generator = Mock()
        mock_generator.generate.return_value = [0.1, 0.2, 0.3]
        
        result = indexer.index_all(mock_generator)
        
        self.assertIsInstance(result, IndexingResult)
        self.assertEqual(result.total_files, 2)
        self.assertGreater(result.total_chunks, 0)
        self.assertEqual(len(result.indexed_files), 2)
        self.assertEqual(len(result.errors), 0)
    
    def test_index_all_saves_index(self):
        """Проверка сохранения индекса."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=50, overlap=5
        )
        
        mock_generator = Mock()
        mock_generator.generate.return_value = [0.1, 0.2, 0.3]
        
        indexer.index_all(mock_generator)
        
        self.assertTrue(os.path.exists(self.embeddings_path))
        
        with open(self.embeddings_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn("chunks", data)
        self.assertGreater(len(data["chunks"]), 0)
    
    def test_index_all_handles_errors(self):
        """Проверка обработки ошибок."""
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=50, overlap=5
        )
        
        # Генератор, который падает на втором файле
        mock_generator = Mock()
        call_count = [0]
        
        def side_effect(text):
            call_count[0] += 1
            if "Second" in text:
                raise Exception("Test error")
            return [0.1, 0.2, 0.3]
        
        mock_generator.generate.side_effect = side_effect
        
        result = indexer.index_all(mock_generator)
        
        # Должна быть одна ошибка
        self.assertEqual(len(result.errors), 1)
        self.assertIn("doc2.txt", result.errors[0])
    
    def test_index_all_empty_directory(self):
        """Проверка индексации пустой директории."""
        empty_dir = os.path.join(self.temp_dir, "empty_docs")
        os.makedirs(empty_dir)
        empty_embeddings = os.path.join(self.temp_dir, "empty_embeddings.json")
        
        indexer = DocumentIndexer(empty_dir, empty_embeddings)
        mock_generator = Mock()
        
        result = indexer.index_all(mock_generator)
        
        self.assertEqual(result.total_files, 0)
        self.assertEqual(result.total_chunks, 0)
        mock_generator.generate.assert_not_called()


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты."""
    
    def setUp(self):
        """Создание временной директории."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.json")
    
    def tearDown(self):
        """Удаление временной директории."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_indexing_workflow(self):
        """Полный workflow индексации."""
        # Создаём документы
        doc1_content = "# Руководство пользователя\n\n" + "А" * 600
        doc2_content = "# FAQ\n\n" + "Б" * 400
        
        with open(os.path.join(self.docs_dir, "guide.md"), 'w', encoding='utf-8') as f:
            f.write(doc1_content)
        with open(os.path.join(self.docs_dir, "faq.md"), 'w', encoding='utf-8') as f:
            f.write(doc2_content)
        
        # Создаём индексатор
        indexer = DocumentIndexer(
            self.docs_dir, self.embeddings_path,
            chunk_size=200, overlap=20
        )
        
        # Мок генератора
        mock_generator = Mock()
        embedding_counter = [0]
        
        def generate_embedding(text):
            embedding_counter[0] += 1
            return [0.1 * embedding_counter[0]] * 10
        
        mock_generator.generate.side_effect = generate_embedding
        
        # Индексируем
        result = indexer.index_all(mock_generator)
        
        # Проверяем результат
        self.assertEqual(result.total_files, 2)
        self.assertGreater(result.total_chunks, 2)  # Должно быть больше 2 чанков
        self.assertEqual(len(result.errors), 0)
        
        # Проверяем сохранённый индекс
        loaded = indexer.load_index()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded["chunks"]), result.total_chunks)
        
        # Проверяем структуру чанка
        first_chunk = loaded["chunks"][0]
        self.assertIn("id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("embedding", first_chunk)
        self.assertIn("source", first_chunk)
        self.assertIn("position", first_chunk)


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование модуля indexer")
    print("=" * 60)
    
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
