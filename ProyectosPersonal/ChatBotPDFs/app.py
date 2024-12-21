import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import spacy
from joblib import dump, load

# Configuración global
MAX_CONTEXTS = 3
MAX_FRAGMENT_SIZE = 300
N_THREADS = 4
VECTOR_CACHE = "tfidf_vectorizer.joblib"
LANGUAGE = 'es_core_news_sm'

# Inicialización de SpaCy
nlp = spacy.load(LANGUAGE)

# Inicialización del modelo
def cargar_modelo():
    print("Cargando el modelo de lenguaje...")
    try:
        return pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            device=0
        )
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Preprocesamiento
def preprocesar_texto(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def dividir_texto(texto, max_longitud=MAX_FRAGMENT_SIZE):
    palabras = texto.split()
    fragmentos = [" ".join(palabras[i:i + max_longitud]) for i in range(0, len(palabras), max_longitud)]
    return fragmentos

# Carga de PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Error al extraer texto del PDF {pdf_path}: {e}")
        return ""

def cargar_documentos(directorio):
    documentos = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.pdf'):
            ruta = os.path.join(directorio, archivo)
            texto = extract_text_from_pdf(ruta)
            if texto.strip():
                documentos.append({"file_name": archivo, "content": texto})
    return documentos

# Contextos relevantes
def buscar_contextos_relevantes(pregunta, pages, vectorizer):
    pregunta_vectorizada = vectorizer.transform([pregunta]).toarray()
    pages_vectorizadas = vectorizer.transform(pages).toarray()
    similitudes = cosine_similarity(pregunta_vectorizada, pages_vectorizadas)[0]
    indices_relevantes = np.argsort(similitudes)[::-1][:MAX_CONTEXTS]
    return [pages[i] for i in indices_relevantes]

# Generar respuesta
def responder_pregunta(pregunta, contextos):
    def procesar_contexto(contexto):
        try:
            return qa_pipeline(question=pregunta, context=contexto)
        except Exception as e:
            print(f"Error procesando contexto: {e}")
            return {"answer": "Error", "score": 0.0}

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        respuestas = list(executor.map(procesar_contexto, contextos))
    return max(respuestas, key=lambda x: x['score'], default={"answer": "No se encontró una respuesta adecuada.", "score": 0.0})

# Inicialización de Flask
app = Flask(__name__)
CORS(app)
qa_pipeline = cargar_modelo()
documents = []
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8)

@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "pregunta" not in data:
            return jsonify({"error": "Se requiere un campo 'pregunta'."}), 400

        pregunta = preprocesar_texto(data["pregunta"])
        if not documents:
            return jsonify({"error": "No hay documentos cargados."}), 400

        contextos = buscar_contextos_relevantes(pregunta, [doc['content'] for doc in documents], vectorizer)
        if not contextos:
            return jsonify({"answer": "No se encontraron contextos relevantes.", "score": 0.0}), 200

        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Error procesando solicitud: {str(e)}"}), 500

if __name__ == "__main__":
    print("Iniciando aplicación...")
    if not qa_pipeline:
        print("Error crítico: No se pudo cargar el modelo.")
        exit(1)

    print("Cargando documentos...")
    documents = cargar_documentos("./PDFs")
    fragmentos = [fragmento for doc in documents for fragmento in dividir_texto(doc['content'])]

    if fragmentos:
        if os.path.exists(VECTOR_CACHE):
            vectorizer = load(VECTOR_CACHE)
        else:
            vectorizer.fit(fragmentos)
            dump(vectorizer, VECTOR_CACHE)
        print(f"{len(documents)} documentos cargados y vectorizados.")
    else:
        print("No se encontraron documentos.")

    app.run(host="0.0.0.0", port=5000, debug=True)