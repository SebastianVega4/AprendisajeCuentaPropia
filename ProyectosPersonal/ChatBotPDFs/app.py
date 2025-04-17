import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import spacy
from joblib import dump, load
from typing import List, Dict, Optional

# Configuración global
class Config:
    MAX_CONTEXTS = 3
    MAX_FRAGMENT_SIZE = 300
    N_THREADS = 4
    VECTOR_CACHE = "tfidf_vectorizer.joblib"
    LANGUAGE = 'es_core_news_sm'
    PDF_DIRECTORY = "./PDFs"
    HOST = "0.0.0.0"
    PORT = 5000

# Inicialización de SpaCy
try:
    nlp = spacy.load(Config.LANGUAGE)
except OSError:
    print(f"Descargando el modelo de lenguaje '{Config.LANGUAGE}'...")
    from spacy.cli import download
    download(Config.LANGUAGE)
    nlp = spacy.load(Config.LANGUAGE)

class DocumentProcessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocesa el texto eliminando stopwords y lematizando."""
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    @staticmethod
    def split_text(text: str, max_length: int = Config.MAX_FRAGMENT_SIZE) -> List[str]:
        """Divide el texto en fragmentos de tamaño máximo especificado."""
        words = text.split()
        return [
            " ".join(words[i:i + max_length]) 
            for i in range(0, len(words), max_length)
        ]

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extrae texto de un archivo PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return " ".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"Error al extraer texto del PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def load_documents(directory: str = Config.PDF_DIRECTORY) -> List[Dict[str, str]]:
        """Carga todos los documentos PDF del directorio especificado."""
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                path = os.path.join(directory, filename)
                text = DocumentProcessor.extract_text_from_pdf(path)
                if text.strip():
                    documents.append({
                        "file_name": filename, 
                        "content": text,
                        "preprocessed": DocumentProcessor.preprocess_text(text)
                    })
        return documents

class QAEngine:
    def __init__(self):
        self.qa_pipeline = self._load_model()
        self.vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8)
        self.documents = []
        self.fragments = []

    def _load_model(self):
        """Carga el modelo de pregunta-respuesta."""
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

    def load_documents(self):
        """Carga y procesa los documentos."""
        print("Cargando documentos...")
        self.documents = DocumentProcessor.load_documents()
        self.fragments = [
            fragment 
            for doc in self.documents 
            for fragment in DocumentProcessor.split_text(doc['content'])
        ]

        if self.fragments:
            if os.path.exists(Config.VECTOR_CACHE):
                self.vectorizer = load(Config.VECTOR_CACHE)
            else:
                self.vectorizer.fit([doc['preprocessed'] for doc in self.documents])
                dump(self.vectorizer, Config.VECTOR_CACHE)
            print(f"{len(self.documents)} documentos cargados y vectorizados.")
        else:
            print("No se encontraron documentos.")

    def find_relevant_contexts(self, question: str) -> List[str]:
        """Encuentra los contextos más relevantes para una pregunta."""
        preprocessed_question = DocumentProcessor.preprocess_text(question)
        question_vector = self.vectorizer.transform([preprocessed_question]).toarray()
        doc_vectors = self.vectorizer.transform(
            [doc['preprocessed'] for doc in self.documents]
        ).toarray()
        
        similarities = cosine_similarity(question_vector, doc_vectors)[0]
        relevant_indices = np.argsort(similarities)[::-1][:Config.MAX_CONTEXTS]
        return [self.documents[i]['content'] for i in relevant_indices]

    def answer_question(self, question: str, contexts: List[str]) -> Dict[str, str]:
        """Genera una respuesta a partir de los contextos dados."""
        def process_context(context: str) -> Dict[str, str]:
            try:
                return self.qa_pipeline(question=question, context=context)
            except Exception as e:
                print(f"Error procesando contexto: {e}")
                return {"answer": "Error", "score": 0.0}

        with ThreadPoolExecutor(max_workers=Config.N_THREADS) as executor:
            answers = list(executor.map(process_context, contexts))
        
        best_answer = max(
            answers, 
            key=lambda x: x['score'], 
            default={"answer": "No se encontró una respuesta adecuada.", "score": 0.0}
        )
        
        # Mejorar la legibilidad de la respuesta
        answer_text = best_answer['answer'].strip()
        if not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
            
        return {
            "answer": answer_text,
            "score": float(best_answer['score']),
            "confidence": self._get_confidence_level(best_answer['score'])
        }

    def _get_confidence_level(self, score: float) -> str:
        """Devuelve un nivel de confianza legible para el usuario."""
        if score > 0.8:
            return "Alta confianza"
        elif score > 0.5:
            return "Media confianza"
        else:
            return "Baja confianza"

# Inicialización de Flask
app = Flask(__name__)
CORS(app)
qa_engine = QAEngine()

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "pregunta" not in data:
            return jsonify({"error": "Se requiere un campo 'pregunta'."}), 400

        question = data["pregunta"].strip()
        if not question:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        if not qa_engine.documents:
            return jsonify({"error": "No hay documentos cargados."}), 400

        contexts = qa_engine.find_relevant_contexts(question)
        if not contexts:
            return jsonify({
                "answer": "No se encontraron contextos relevantes.", 
                "score": 0.0,
                "confidence": "N/A"
            }), 200

        response = qa_engine.answer_question(question, contexts)
        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "error": f"Error procesando solicitud: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("Iniciando aplicación...")
    
    if not qa_engine.qa_pipeline:
        print("Error crítico: No se pudo cargar el modelo.")
        exit(1)

    qa_engine.load_documents()
    
    app.run(
        host=Config.HOST, 
        port=Config.PORT, 
        debug=True
    )