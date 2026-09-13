"""Общий RAG-модуль: чанкинг документации, эмбеддинги, FAISS-индекс и поиск.

Импортируется из ноутбуков:

    from rag import build_index, vector_search
    rag = build_index("docs")
    vector_search(rag, "What is a Modelfile?", top_k=3)
"""

from dataclasses import dataclass
from pathlib import Path
import requests

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCS_DIR = Path("docs")
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:1.5b"
PROMPT_TEMPLATE = """Ты ассистент по документации Ollama.
Отвечай на вопрос пользователя, опираясь только на приведённые ниже фрагменты документации.
Если ответа во фрагментах нет - честно скажи, что не знаешь.
Ответ давай на русском языке.
В конце ответа на отдельной строке укажи источник в формате: Источник: <имя файла>.

Фрагменты документации:
{context}

Вопрос пользователя: {question}

Ответ:"""


def chunk_text(text: str, max_chars: int = 800, min_chars: int = 50) -> list[str]:
    """Режет Markdown по заголовкам ##; длинные секции дробит по параграфам.

    Параметры:
        text: исходный Markdown-текст
        max_chars: максимальная длина одного чанка в символах
        min_chars: минимальная длина чанка - чанки короче выбрасываются

    Возвращает:
        Список текстовых чанков
    """
    lines = text.split("\n")
    sections, current = [], []
    for line in lines:
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks = []
    for section in sections:
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        # Длинную секцию дробим по двойным переносам (параграфам)
        buf = ""
        for paragraph in section.split("\n\n"):
            if len(buf) + len(paragraph) + 2 <= max_chars:
                buf = f"{buf}\n\n{paragraph}" if buf else paragraph
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = paragraph
        if buf:
            chunks.append(buf.strip())

    return [c for c in chunks if len(c) >= min_chars]


def load_chunks(docs_dir: str | Path = DOCS_DIR) -> list[dict]:
    """Читает все .md/.mdx из папки и режет их на чанки с указанием источника."""
    docs_dir = Path(docs_dir)
    chunks = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() in {".md", ".mdx"}:
            text = path.read_text(encoding="utf-8")
            for chunk in chunk_text(text):
                chunks.append({"text": chunk, "source": str(path)})
    return chunks


def load_embed_model(name: str = EMBED_MODEL) -> SentenceTransformer:
    """Загружает модель эмбеддингов (при первом запуске качает с Hugging Face)."""
    return SentenceTransformer(name)


def embed_texts(
        texts: list[str],
        embed_model: SentenceTransformer,
        show_progress_bar: bool = False,
) -> np.ndarray:
    """Считает нормализованные эмбеддинги: матрица (N, 384) типа float32."""
    return embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    ).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Плоский индекс по скалярному произведению (= косинус для нормализованных векторов)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


@dataclass
class RagIndex:
    """Всё, что нужно для поиска: чанки, модель и FAISS-индекс."""

    chunks: list[dict]
    embed_model: SentenceTransformer
    index: faiss.Index


def build_index(docs_dir: str | Path = DOCS_DIR, show_progress_bar: bool = False) -> RagIndex:
    """Полный пайплайн: документы -> чанки -> эмбеддинги -> индекс."""
    chunks = load_chunks(docs_dir)
    embed_model = load_embed_model()
    embeddings = embed_texts([c["text"] for c in chunks], embed_model, show_progress_bar)
    index = build_faiss_index(embeddings)
    return RagIndex(chunks=chunks, embed_model=embed_model, index=index)


def vector_search(rag: RagIndex, question: str, top_k: int = 3) -> list[dict]:
    """Возвращает top_k чанков, наиболее близких к запросу по смыслу."""
    # 1. Считаем эмбеддинг запроса той же моделью, что и чанки
    query_emb = embed_texts([question], rag.embed_model)

    # 2. Ищем ближайших соседей
    scores, indices = rag.index.search(query_emb, top_k)

    # 3. Собираем результат с метаданными
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": rag.chunks[idx]["text"],
            "source": rag.chunks[idx]["source"],
            "score": float(score),
        })
    return results


def generate_answer(prompt: str) -> str:
    """Отправляет промпт в Ollama и возвращает сгенерированный ответ."""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    return data["response"].strip()


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Собирает промпт из системной инструкции, контекста и вопроса."""
    context_parts = []
    for r in retrieved:
        context_parts.append(f"[файл: {r['source']}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)
