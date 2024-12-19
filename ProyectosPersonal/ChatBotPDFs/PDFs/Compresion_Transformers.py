import numpy as np
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def responder_pregunta(pregunta, contextos):
    respuestas = []
    for contexto in contextos:
        respuesta = qa_pipeline(question=pregunta, context=contexto)
        respuestas.append(respuesta)
    return max(respuestas, key=lambda x: x['score'])

def buscar_contextos_relevantes(pregunta, pages, index, vectorizer):
    pregunta_vectorizada = vectorizer.transform([pregunta]).toarray()
    _, indices = index.search(np.array(pregunta_vectorizada, dtype=np.float32), k=5)

    return [pages[i] for i in indices[0]]