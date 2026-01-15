"""
RAG (Retrieval-Augmented Generation) модуль.

Компоненты:
    - indexer: Индексация документов из директории docs/
    - embeddings: Генерация эмбедингов через локальную LLM
    - retriever: Поиск релевантных чанков по косинусному сходству
"""

from .indexer import DocumentIndexer
from .embeddings import EmbeddingGenerator
from .retriever import DocumentRetriever

__all__ = ["DocumentIndexer", "EmbeddingGenerator", "DocumentRetriever"]
