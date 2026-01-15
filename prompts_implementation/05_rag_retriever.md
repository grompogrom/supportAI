# Промпт для имплементации: RAG Retriever

## Задача
Реализовать поиск релевантных документов по косинусному сходству.

## Файл
`src/rag/retriever.py`

## Контекст
Этот модуль отвечает за поиск наиболее релевантных чанков документации по запросу пользователя. Используется косинусное сходство между эмбедингами.

## Требования к имплементации

### Класс DocumentRetriever

#### `__init__(self, embeddings_path: str, embedding_generator: 'EmbeddingGenerator')`
```
Реализуй инициализацию:
1. Сохрани путь: self._embeddings_path = embeddings_path
2. Сохрани генератор: self._embedding_generator = embedding_generator
3. Инициализируй индекс как None: self._index = None
4. Сохрани время модификации файла: self._index_mtime = None
```

#### `search(self, query: str, top_k: int = 3) -> List[SearchResult]`
```
Реализуй поиск:
1. Проверь и загрузи индекс:
   self._reload_index_if_needed()
   if self._index is None:
       raise IndexNotFoundError("Индекс не найден. Запустите /index")

2. Сгенерируй эмбединг запроса:
   query_embedding = self._embedding_generator.generate(query)

3. Вычисли сходство со всеми чанками:
   similarities = self._compute_all_similarities(query_embedding)

4. Отсортируй по убыванию сходства:
   similarities.sort(key=lambda x: x[1], reverse=True)

5. Возьми топ-K результатов:
   top_results = similarities[:top_k]

6. Сформируй SearchResult для каждого:
   results = []
   for idx, score in top_results:
       chunk = self._index["chunks"][idx]
       results.append(SearchResult(
           chunk_id=chunk["id"],
           text=chunk["text"],
           source_file=chunk["source"],
           similarity_score=score,
           position=chunk["position"]
       ))
   return results
```

#### `load_index(self) -> bool`
```
Загрузи индекс в память:
1. Проверь существование файла:
   if not os.path.exists(self._embeddings_path):
       return False
2. Загрузи JSON:
   with open(self._embeddings_path, 'r', encoding='utf-8') as f:
       self._index = json.load(f)
3. Проверь наличие чанков:
   if not self._index.get("chunks"):
       raise IndexCorruptedError("Индекс пуст или поврежден")
4. Сохрани время модификации:
   self._index_mtime = os.path.getmtime(self._embeddings_path)
5. return True
```

#### `is_index_loaded(self) -> bool`
```
return self._index is not None
```

#### `get_index_stats(self) -> dict`
```
Верни статистику по индексу:
1. Если индекс не загружен - загрузи
2. Собери статистику:
   - total_chunks: количество чанков
   - indexed_at: дата индексации
   - source_files: уникальные исходные файлы
   
return {
    "total_chunks": len(self._index["chunks"]),
    "indexed_at": self._index.get("indexed_at"),
    "source_files": list(set(c["source"] for c in self._index["chunks"])),
    "config": self._index.get("config", {})
}
```

#### `format_results_for_llm(self, results: List[SearchResult]) -> str`
```
Сформируй текст для LLM:

output = "Найденные документы:\n\n"
for i, result in enumerate(results, 1):
    output += f"[{i}] Источник: {result.source_file} "
    output += f"(релевантность: {result.similarity_score:.2f})\n"
    output += f"{result.text}\n\n"
return output.strip()
```

#### `_compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float`
```
Вычисли косинусное сходство:

Формула: cos(θ) = (A · B) / (||A|| * ||B||)

Реализация с numpy:
import numpy as np

vec1 = np.array(vec1)
vec2 = np.array(vec2)

dot_product = np.dot(vec1, vec2)
norm1 = np.linalg.norm(vec1)
norm2 = np.linalg.norm(vec2)

if norm1 == 0 or norm2 == 0:
    return 0.0

return dot_product / (norm1 * norm2)

Альтернативная реализация без numpy:
import math

dot_product = sum(a * b for a, b in zip(vec1, vec2))
norm1 = math.sqrt(sum(a * a for a in vec1))
norm2 = math.sqrt(sum(b * b for b in vec2))

if norm1 == 0 or norm2 == 0:
    return 0.0

return dot_product / (norm1 * norm2)
```

#### `_compute_all_similarities(self, query_embedding: List[float]) -> List[Tuple[int, float]]`
```
Вычисли сходство со всеми чанками:

similarities = []
for idx, chunk in enumerate(self._index["chunks"]):
    similarity = self._compute_cosine_similarity(
        query_embedding,
        chunk["embedding"]
    )
    similarities.append((idx, similarity))
return similarities
```

#### `_reload_index_if_needed(self)`
```
Перезагрузи индекс если файл изменился:

1. Если индекс не загружен - загрузи:
   if self._index is None:
       self.load_index()
       return

2. Проверь время модификации файла:
   if os.path.exists(self._embeddings_path):
       current_mtime = os.path.getmtime(self._embeddings_path)
       if current_mtime != self._index_mtime:
           print("Индекс обновлен, перезагружаем...")
           self.load_index()
```

## Формула косинусного сходства

```
         A · B           Σ(Ai × Bi)
cos(θ) = ───── = ─────────────────────────
        ||A|| × ||B||   √Σ(Ai²) × √Σ(Bi²)
```

Где:
- A · B - скалярное произведение векторов
- ||A||, ||B|| - нормы (длины) векторов

Значение от -1 до 1:
- 1 = идентичные направления
- 0 = ортогональные (не связаны)
- -1 = противоположные направления

Для эмбедингов обычно значения от 0 до 1.

## Пример работы

```python
retriever = DocumentRetriever("data/embeddings.json", embedding_generator)
results = retriever.search("как сбросить пароль", top_k=3)

for r in results:
    print(f"[{r.similarity_score:.2f}] {r.source_file}")
    print(f"  {r.text[:100]}...")
```

## Зависимости
- numpy (или math для реализации без numpy)
- os
- json
- typing

## Тестирование
После реализации проверь:
1. Корректный расчет косинусного сходства
2. Правильную сортировку результатов
3. Форматирование вывода для LLM
4. Автоматическую перезагрузку индекса
