import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline

# Función para inicializar el modelo
def cargar_modelo():
    """
    Inicializa el pipeline de Transformers para la tarea de Question Answering.
    """
    try:
        print("Cargando el modelo...")
        qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
        print("Modelo cargado exitosamente.")
        return qa_pipeline
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Función para responder preguntas
def responder_pregunta(pregunta, contextos):
    """
    Genera respuestas para una pregunta dada utilizando los contextos relevantes.
    """
    respuestas = []
    for contexto in contextos:
        try:
            respuesta = qa_pipeline(question=pregunta, context=contexto)
            respuestas.append(respuesta)
        except Exception as e:
            print(f"Error al procesar la pregunta: {e}")
    if respuestas:
        return max(respuestas, key=lambda x: x['score'])
    return {"answer": "No se encontró una respuesta adecuada.", "score": 0.0}

# Función para buscar contextos relevantes usando TF-IDF
def buscar_contextos_relevantes(pregunta, pages, vectorizer):
    """
    Busca los contextos más relevantes para una pregunta usando similitud coseno y TF-IDF.
    """
    pregunta_vectorizada = vectorizer.transform([pregunta]).toarray()
    pages_vectorizadas = vectorizer.transform(pages).toarray()
    similitudes = cosine_similarity(pregunta_vectorizada, pages_vectorizadas)[0]
    indices_relevantes = np.argsort(similitudes)[::-1][:5]
    return [pages[i] for i in indices_relevantes]

# Función para extraer texto de PDFs
def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texto = " ".join(page.extract_text() or "" for page in pdf.pages)
            if not texto.strip():
                print(f"Advertencia: No se extrajo texto del archivo {pdf_path}.")
            return texto
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return ""

# Cargar PDFs desde la carpeta
def load_pdfs_from_directory(directory):
    """
    Carga y procesa todos los archivos PDF en un directorio.
    """
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
qa_pipeline = None
documents = []
vectorizer = TfidfVectorizer()

@app.route("/")
def home():
    """
    Ruta principal para servir el archivo HTML de interfaz.
    """
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint para manejar preguntas y devolver respuestas.
    """
    try:
        data = request.get_json()
        if not data or "pregunta" not in data:
            return jsonify({"error": "Entrada no válida. Se requiere un campo 'pregunta'."}), 400

        pregunta = data["pregunta"].strip()
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        if not documents:
            return jsonify({"error": "No hay datos cargados"}), 400

        contextos = buscar_contextos_relevantes(pregunta, [doc['content'] for doc in documents], vectorizer)
        if not contextos:
            return jsonify({"answer": "No se encontraron contextos relevantes.", "score": 0.0}), 200

        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    # Inicializa documentos y modelo en el bloque principal
    print("Iniciando aplicación...")

    # Cargar modelo
    qa_pipeline = cargar_modelo()

    if not qa_pipeline:
        print("Error crítico: No se pudo cargar el modelo. Saliendo.")
        exit(1)

    # Cargar documentos
    print("Cargando documentos desde la carpeta PDFs...")
    documents = load_pdfs_from_directory("./PDFs")
    if documents:
        print(f"{len(documents)} documentos cargados exitosamente.")
        vectorizer.fit([doc['content'] for doc in documents])
    else:
        print("No se encontraron documentos en la carpeta PDFs. El servidor se ejecutará, pero no podrá responder preguntas.")

    # Iniciar aplicación Flask
    app.run(host="0.0.0.0", port=5000, debug=True)