"""
Тесты для модуля генерации эмбедингов.
"""

import unittest
from unittest.mock import patch, Mock
import sys
import os

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.embeddings import (
    EmbeddingConfig,
    EmbeddingGenerator,
    EmbeddingError,
    EmbeddingConnectionError,
    EmbeddingParseError,
)


class TestEmbeddingConfig(unittest.TestCase):
    """Тесты для EmbeddingConfig."""
    
    def test_config_creation(self):
        """Проверка создания конфигурации."""
        config = EmbeddingConfig(
            host="localhost",
            port=11434,
            model_name="mxbai-embed-large",
            endpoint="/api/embeddings"
        )
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 11434)
        self.assertEqual(config.model_name, "mxbai-embed-large")
        self.assertEqual(config.endpoint, "/api/embeddings")
        self.assertEqual(config.timeout, 30)  # default
        self.assertEqual(config.retry_attempts, 3)  # default
    
    def test_config_with_custom_timeout(self):
        """Проверка создания конфигурации с кастомным таймаутом."""
        config = EmbeddingConfig(
            host="localhost",
            port=11434,
            model_name="mxbai-embed-large",
            endpoint="/api/embeddings",
            timeout=60,
            retry_attempts=5
        )
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.retry_attempts, 5)


class TestEmbeddingGeneratorUnit(unittest.TestCase):
    """Unit-тесты для EmbeddingGenerator (с моками)."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.config = EmbeddingConfig(
            host="localhost",
            port=11434,
            model_name="mxbai-embed-large",
            endpoint="/api/embeddings"
        )
        self.generator = EmbeddingGenerator(self.config)
    
    def test_init(self):
        """Проверка инициализации генератора."""
        self.assertEqual(self.generator._config, self.config)
        self.assertEqual(
            self.generator._base_url,
            "http://localhost:11434/api/embeddings"
        )
        self.assertIsNone(self.generator._embedding_dim)
    
    @patch('rag.embeddings.requests.post')
    def test_send_request_success(self, mock_post):
        """Проверка успешного запроса."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        result = self.generator._send_request("test text")
        
        self.assertEqual(result, {"embedding": [0.1, 0.2, 0.3]})
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": "test text"},
            timeout=30
        )
    
    @patch('rag.embeddings.requests.post')
    def test_send_request_connection_error(self, mock_post):
        """Проверка обработки ошибки подключения."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        with self.assertRaises(EmbeddingConnectionError) as context:
            self.generator._send_request("test text")
        
        self.assertIn("Не удалось подключиться", str(context.exception))
    
    @patch('rag.embeddings.requests.post')
    def test_send_request_timeout(self, mock_post):
        """Проверка обработки таймаута."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with self.assertRaises(EmbeddingConnectionError) as context:
            self.generator._send_request("test text")
        
        self.assertIn("Таймаут", str(context.exception))
    
    @patch('rag.embeddings.requests.post')
    def test_send_request_api_error(self, mock_post):
        """Проверка обработки ошибки API."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with self.assertRaises(EmbeddingConnectionError) as context:
            self.generator._send_request("test text")
        
        self.assertIn("500", str(context.exception))
    
    def test_parse_embedding_success(self):
        """Проверка успешного парсинга эмбединга."""
        response = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        result = self.generator._parse_embedding(response)
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5])
    
    def test_parse_embedding_missing_field(self):
        """Проверка обработки отсутствующего поля."""
        response = {"data": [0.1, 0.2, 0.3]}
        
        with self.assertRaises(EmbeddingParseError) as context:
            self.generator._parse_embedding(response)
        
        self.assertIn("embedding", str(context.exception))
    
    @patch('rag.embeddings.requests.post')
    def test_generate_success(self, mock_post):
        """Проверка успешной генерации эмбединга."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        result = self.generator.generate("test text")
        
        self.assertEqual(result, [0.1, 0.2, 0.3])
    
    @patch('rag.embeddings.requests.post')
    def test_generate_batch(self, mock_post):
        """Проверка батчевой генерации."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
            {"embedding": [0.5, 0.6]},
        ]
        mock_post.return_value = mock_response
        
        result = self.generator.generate_batch(["text1", "text2", "text3"])
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [0.1, 0.2])
        self.assertEqual(result[1], [0.3, 0.4])
        self.assertEqual(result[2], [0.5, 0.6])
    
    @patch('rag.embeddings.requests.post')
    def test_check_model_availability_true(self, mock_post):
        """Проверка доступности модели - успех."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1]}
        mock_post.return_value = mock_response
        
        result = self.generator.check_model_availability()
        
        self.assertTrue(result)
    
    @patch('rag.embeddings.requests.post')
    def test_check_model_availability_false(self, mock_post):
        """Проверка доступности модели - неудача."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        result = self.generator.check_model_availability()
        
        self.assertFalse(result)
    
    @patch('rag.embeddings.requests.post')
    def test_get_embedding_dimension(self, mock_post):
        """Проверка получения размерности эмбединга."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 1024}
        mock_post.return_value = mock_response
        
        dim = self.generator.get_embedding_dimension()
        
        self.assertEqual(dim, 1024)
        self.assertEqual(self.generator._embedding_dim, 1024)
        
        # Второй вызов должен использовать кэш
        dim2 = self.generator.get_embedding_dimension()
        self.assertEqual(dim2, 1024)
        # Проверяем, что запрос был только один раз
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('rag.embeddings.time.sleep')
    @patch('rag.embeddings.requests.post')
    def test_retry_with_backoff(self, mock_post, mock_sleep):
        """Проверка retry с экспоненциальным backoff."""
        import requests
        
        # Первые 2 попытки неудачны, третья успешна
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1]}
        
        mock_post.side_effect = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectionError(),
            mock_response,
        ]
        
        result = self.generator.generate("test")
        
        self.assertEqual(result, [0.1])
        self.assertEqual(mock_post.call_count, 3)
        # Проверяем вызовы sleep с правильными задержками
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1)  # 2^0 = 1
        mock_sleep.assert_any_call(2)  # 2^1 = 2


class TestEmbeddingGeneratorIntegration(unittest.TestCase):
    """
    Интеграционные тесты для EmbeddingGenerator.
    Требуют запущенный Ollama с моделью mxbai-embed-large.
    """
    
    @classmethod
    def setUpClass(cls):
        """Настройка перед всеми тестами."""
        cls.config = EmbeddingConfig(
            host="localhost",
            port=11434,
            model_name="mxbai-embed-large",
            endpoint="/api/embeddings"
        )
        cls.generator = EmbeddingGenerator(cls.config)
        
        # Проверяем доступность Ollama
        cls.ollama_available = cls.generator.check_model_availability()
        if not cls.ollama_available:
            print("\n⚠️  Ollama недоступен. Интеграционные тесты будут пропущены.")
    
    def setUp(self):
        """Пропуск тестов если Ollama недоступен."""
        if not self.ollama_available:
            self.skipTest("Ollama недоступен")
    
    def test_generate_real_embedding(self):
        """Тест генерации реального эмбединга."""
        embedding = self.generator.generate("Привет мир")
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertTrue(all(isinstance(x, float) for x in embedding))
        
        print(f"\n✓ Размерность эмбединга: {len(embedding)}")
        print(f"✓ Первые 5 значений: {embedding[:5]}")
    
    def test_generate_batch_real(self):
        """Тест батчевой генерации."""
        texts = ["Первый текст", "Второй текст", "Третий текст"]
        embeddings = self.generator.generate_batch(texts)
        
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertIsInstance(emb, list)
            self.assertGreater(len(emb), 0)
        
        # Все эмбединги должны иметь одинаковую размерность
        dims = [len(e) for e in embeddings]
        self.assertEqual(len(set(dims)), 1)
        
        print(f"\n✓ Сгенерировано {len(embeddings)} эмбедингов")
        print(f"✓ Размерность: {dims[0]}")
    
    def test_get_embedding_dimension_real(self):
        """Тест получения размерности."""
        dim = self.generator.get_embedding_dimension()
        
        self.assertIsInstance(dim, int)
        self.assertGreater(dim, 0)
        
        print(f"\n✓ Размерность модели mxbai-embed-large: {dim}")
    
    def test_different_texts_different_embeddings(self):
        """Проверка, что разные тексты дают разные эмбединги."""
        emb1 = self.generator.generate("Кошка сидит на окне")
        emb2 = self.generator.generate("Собака бежит по улице")
        
        # Эмбединги не должны быть идентичными
        self.assertNotEqual(emb1, emb2)
        
        # Вычисляем косинусное сходство
        import math
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        similarity = dot_product / (norm1 * norm2)
        
        print(f"\n✓ Косинусное сходство между разными текстами: {similarity:.4f}")
        self.assertLess(similarity, 0.99)  # Должны отличаться
    
    def test_similar_texts_similar_embeddings(self):
        """Проверка, что похожие тексты дают похожие эмбединги."""
        emb1 = self.generator.generate("Кошка сидит на окне")
        emb2 = self.generator.generate("Кот сидит у окна")
        
        # Вычисляем косинусное сходство
        import math
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        similarity = dot_product / (norm1 * norm2)
        
        print(f"\n✓ Косинусное сходство между похожими текстами: {similarity:.4f}")
        self.assertGreater(similarity, 0.7)  # Должны быть похожи


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование модуля embeddings")
    print("Модель: mxbai-embed-large")
    print("=" * 60)
    
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
