import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline

def cargar_modelo():
    """
    Inicializa el modelo con el pipeline de Transformers.
    """
    try:
        print("Cargando el modelo...")
        return pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            device=-1  # Asegura el uso de la CPU
        )
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def dividir_texto(texto, max_longitud=500):
    """
    Divide el texto en fragmentos más pequeños para facilitar el procesamiento.
    """
    palabras = texto.split()
    fragmentos = []
    for i in range(0, len(palabras), max_longitud):
        fragmentos.append(" ".join(palabras[i:i + max_longitud]))
    return fragmentos

def responder_pregunta(pregunta, contextos):
    """
    Genera respuestas usando el pipeline y los contextos relevantes.
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

def buscar_contextos_relevantes(pregunta, pages, vectorizer):
    """
    Busca contextos relevantes utilizando TF-IDF.
    """
    pregunta_vectorizada = vectorizer.transform([pregunta]).toarray()
    pages_vectorizadas = vectorizer.transform(pages).toarray()
    similitudes = cosine_similarity(pregunta_vectorizada, pages_vectorizadas)[0]
    indices_relevantes = np.argsort(similitudes)[::-1][:5]
    return [pages[i] for i in indices_relevantes]

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un PDF.
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

def load_pdfs_from_directory(directory):
    """
    Carga documentos PDF desde un directorio.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            full_path = os.path.join(directory, file_name)
            text = extract_text_from_pdf(full_path)
            if text:
                documents.append({"file_name": file_name, "content": text})
    return documents

app = Flask(__name__)
CORS(app)  # Configuración de CORS
qa_pipeline = None
documents = []
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8)

@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Obtener datos JSON de la solicitud
        data = request.get_json()
        print(f"Datos recibidos: {data}")  # Log para depuración

        # Validar que el JSON contiene el campo 'pregunta'
        if not data or "pregunta" not in data:
            return jsonify({"error": "Entrada no válida. Se requiere un campo 'pregunta'."}), 400

        # Validar que la pregunta no esté vacía
        pregunta = data["pregunta"].strip()
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        # Verificar si hay documentos cargados
        if not documents:
            return jsonify({"error": "No hay datos cargados."}), 400

        # Buscar contextos relevantes
        contextos = buscar_contextos_relevantes(pregunta, [doc['content'] for doc in documents], vectorizer)
        if not contextos:
            return jsonify({"answer": "No se encontraron contextos relevantes.", "score": 0.0}), 200

        # Generar respuesta con el modelo
        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']}), 200

    except Exception as e:
        # Manejar errores inesperados
        print(f"Error: {e}")
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    print("Iniciando aplicación...")
    qa_pipeline = cargar_modelo()
    if not qa_pipeline:
        print("Error crítico: No se pudo cargar el modelo.")
        exit(1)

    print("Cargando documentos desde la carpeta PDFs...")
    documents = load_pdfs_from_directory("./PDFs")
    fragmentos = []
    for doc in documents:
        fragmentos.extend(dividir_texto(doc['content']))
    if fragmentos:
        vectorizer.fit(fragmentos)
        print(f"{len(documents)} documentos cargados exitosamente.")
    else:
        print("No se encontraron documentos.")

    app.run(host="0.0.0.0", port=5000, debug=True)