import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber
from flask import Flask, request, jsonify
from transformers import pipeline

# Inicializar pipeline de Transformers
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

# Función para responder preguntas
def responder_pregunta(pregunta, contextos):
    respuestas = []
    for contexto in contextos:
        respuesta = qa_pipeline(question=pregunta, context=contexto)
        respuestas.append(respuesta)
    return max(respuestas, key=lambda x: x['score'])

# Función para buscar contextos relevantes usando TF-IDF
def buscar_contextos_relevantes(pregunta, pages, vectorizer):
    pregunta_vectorizada = vectorizer.transform([pregunta]).toarray()
    pages_vectorizadas = vectorizer.transform(pages).toarray()
    similitudes = cosine_similarity(pregunta_vectorizada, pages_vectorizadas)[0]
    indices_relevantes = np.argsort(similitudes)[::-1][:5]
    return [pages[i] for i in indices_relevantes]

# Función para extraer texto de PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return ""

# Cargar PDFs desde la carpeta
def load_pdfs_from_directory(directory):
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            full_path = os.path.join(directory, file_name)
            text = extract_text_from_pdf(full_path)
            if text:
                documents.append({"file_name": file_name, "content": text})
    return documents

# Inicializar Flask
app = Flask(__name__)

# Variables globales
documents = []
vectorizer = TfidfVectorizer()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        pregunta = request.json.get("pregunta")
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        if not documents:
            return jsonify({"error": "No hay datos cargados"}), 400

        contextos = buscar_contextos_relevantes(pregunta, [doc['content'] for doc in documents], vectorizer)
        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']}), 200
    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {e}"}), 500

if __name__ == "__main__":
    print("Cargando documentos desde la carpeta PDFs...")
    documents = load_pdfs_from_directory("./PDFs")
    if documents:
        print(f"{len(documents)} documentos cargados exitosamente.")
        vectorizer.fit([doc['content'] for doc in documents])
    else:
        print("No se encontraron documentos en la carpeta PDFs.")
    app.run(host="0.0.0.0", port=5000, debug=True)