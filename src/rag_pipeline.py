
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from src.pdf_loader import extract_text
from src.chunking import split_text

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

documents_store = []

def process_documents(files):
    global documents_store

    texts = []
    metadata = []

    for file in files:
        text = extract_text(file)

        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            texts.append(chunk)

            metadata.append({
                "source": file.name,
                "chunk": i + 1
            })

    embeddings = embedding_model.encode(texts)

    documents_store = list(zip(texts, metadata))

    return {
        "embeddings": embeddings,
        "documents": documents_store
    }

def ask_question(index, question):
    question_embedding = embedding_model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        index["embeddings"]
    )

    top_indices = similarities.argsort()[0][-3:][::-1]

    retrieved_chunks = []

    sources = []

    for idx in top_indices:
        chunk, meta = index["documents"][idx]

        retrieved_chunks.append(chunk)

        sources.append(
            f"{meta['source']} - Chunk {meta['chunk']}"
        )

    context = "\\n".join(retrieved_chunks)

    prompt = f'''
    Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {question}
    '''

    result = qa_pipeline(
        prompt,
        max_length=256
    )[0]["generated_text"]

    return {
        "answer": result,
        "sources": sources
    }