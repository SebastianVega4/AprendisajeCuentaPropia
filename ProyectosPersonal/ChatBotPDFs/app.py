import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from docx import Document
import pickle
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# Configuración global
class Config:
    MAX_CONTEXTS = 5  # Aumentado para mejor precisión
    MAX_FRAGMENT_SIZE = 500  # Fragmentos más grandes para mejor contexto
    N_THREADS = 4
    VECTOR_CACHE = "vector_cache.pkl"
    DOCUMENT_CACHE = "doc_cache.pkl"
    UPLOAD_FOLDER = "./uploads"
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    HOST = "0.0.0.0"
    PORT = 5000
    MODEL_NAME = "mrm8488/distillbert-base-spanish-wwm-cased-finetuned-spa-squad2-es"  # Modelo más ligero

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Inicialización de Flask
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
Path(Config.UPLOAD_FOLDER).mkdir(exist_ok=True)

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        self.punctuation = re.compile(r'[^\w\s]')

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto eliminando stopwords y aplicando stemming."""
        # Limpieza básica
        text = text.lower()
        text = self.punctuation.sub(' ', text)
        
        # Tokenización y stemming
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def split_text(self, text: str, max_length: int = Config.MAX_FRAGMENT_SIZE) -> List[str]:
        """Divide el texto en fragmentos manteniendo párrafos intactos cuando es posible."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        fragments = []
        current_fragment = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para.split())
            if current_length + para_length <= max_length:
                current_fragment.append(para)
                current_length += para_length
            else:
                if current_fragment:
                    fragments.append('\n'.join(current_fragment))
                current_fragment = [para]
                current_length = para_length

        if current_fragment:
            fragments.append('\n'.join(current_fragment))

        return fragments

class DocumentHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.85)
        self.documents = []
        self.fragments = []
        self.document_hashes = set()

    def _file_hash(self, file_path: str) -> str:
        """Calcula el hash de un archivo para detectar cambios."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def extract_text(self, file_path: str) -> str:
        """Extrae texto de diferentes tipos de archivos."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        elif ext == '.docx':
            doc = Document(file_path)
            return '\n'.join(para.text for para in doc.paragraphs if para.text.strip())
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Formato de archivo no soportado: {ext}")

    def load_documents(self, directory: str = Config.UPLOAD_FOLDER) -> bool:
        """Carga y procesa documentos del directorio especificado."""
        try:
            # Verificar si hay documentos nuevos o modificados
            current_hashes = {}
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
                    file_path = os.path.join(directory, filename)
                    current_hashes[filename] = self._file_hash(file_path)

            # Si no hay cambios, cargar de caché
            if (os.path.exists(Config.DOCUMENT_CACHE) and 
                os.path.exists(Config.VECTOR_CACHE) and
                current_hashes == self.document_hashes):
                with open(Config.DOCUMENT_CACHE, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(Config.VECTOR_CACHE, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.fragments = [f for doc in self.documents for f in doc['fragments']]
                return True

            # Procesar documentos
            self.documents = []
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
                    file_path = os.path.join(directory, filename)
                    try:
                        text = self.extract_text(file_path)
                        if text.strip():
                            preprocessed = self.text_processor.preprocess_text(text)
                            fragments = self.text_processor.split_text(text)
                            self.documents.append({
                                'file_name': filename,
                                'content': text,
                                'preprocessed': preprocessed,
                                'fragments': fragments
                            })
                    except Exception as e:
                        print(f"Error procesando {filename}: {str(e)}")

            if not self.documents:
                return False

            # Actualizar hashes y guardar en caché
            self.document_hashes = current_hashes
            self.fragments = [f for doc in self.documents for f in doc['fragments']]
            
            # Entrenar vectorizador
            self.vectorizer.fit([doc['preprocessed'] for doc in self.documents])
            
            # Guardar en caché
            with open(Config.DOCUMENT_CACHE, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(Config.VECTOR_CACHE, 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            return True
        except Exception as e:
            print(f"Error al cargar documentos: {str(e)}")
            return False

    def find_relevant_contexts(self, question: str) -> List[str]:
        """Encuentra los contextos más relevantes para una pregunta."""
        if not self.documents:
            return []

        preprocessed_question = self.text_processor.preprocess_text(question)
        question_vector = self.vectorizer.transform([preprocessed_question]).toarray()
        
        doc_vectors = self.vectorizer.transform(
            [doc['preprocessed'] for doc in self.documents]
        ).toarray()
        
        similarities = cosine_similarity(question_vector, doc_vectors)[0]
        relevant_indices = np.argsort(similarities)[::-1][:Config.MAX_CONTEXTS]
        
        # Seleccionar los fragmentos más relevantes de los documentos más relevantes
        contexts = []
        for idx in relevant_indices:
            doc = self.documents[idx]
            fragment_vectors = self.vectorizer.transform(
                [self.text_processor.preprocess_text(f) for f in doc['fragments']]
            ).toarray()
            
            frag_similarities = cosine_similarity(question_vector, fragment_vectors)[0]
            best_frag_idx = np.argmax(frag_similarities)
            contexts.append(doc['fragments'][best_frag_idx])
        
        return contexts

class QAEngine:
    def __init__(self):
        self.qa_pipeline = None
        self.document_handler = DocumentHandler()
        self.load_model()
        
    def load_model(self):
        """Carga el modelo de pregunta-respuesta."""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=Config.MODEL_NAME,
                tokenizer=Config.MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1
            )
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False

    def answer_question(self, question: str) -> Dict[str, str]:
        """Genera una respuesta a partir de los documentos cargados."""
        if not self.document_handler.documents:
            return {
                "answer": "No hay documentos cargados. Por favor, sube archivos primero.",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }

        contexts = self.document_handler.find_relevant_contexts(question)
        if not contexts:
            return {
                "answer": "No se encontraron contextos relevantes para tu pregunta.",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }

        def process_context(context: str) -> Dict:
            try:
                result = self.qa_pipeline(question=question, context=context, max_seq_len=512)
                return {
                    "answer": result['answer'],
                    "score": float(result['score']),
                    "context": context
                }
            except Exception as e:
                print(f"Error procesando contexto: {str(e)}")
                return {
                    "answer": "",
                    "score": 0.0,
                    "context": context
                }

        with ThreadPoolExecutor(max_workers=Config.N_THREADS) as executor:
            answers = list(executor.map(process_context, contexts))
        
        # Filtrar respuestas vacías y seleccionar la mejor
        valid_answers = [a for a in answers if a['answer'].strip()]
        if not valid_answers:
            return {
                "answer": "No pude encontrar una respuesta en los documentos.",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }

        best_answer = max(valid_answers, key=lambda x: x['score'])
        
        # Formatear respuesta
        answer_text = best_answer['answer'].strip()
        if not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
        
        # Obtener fuentes (nombres de documentos)
        sources = []
        for doc in self.document_handler.documents:
            if best_answer['context'] in doc['fragments']:
                sources.append(doc['file_name'])
                if len(sources) >= 2:  # Limitar a 2 fuentes
                    break
        
        return {
            "answer": answer_text,
            "score": best_answer['score'],
            "confidence": self._get_confidence_level(best_answer['score']),
            "sources": sources
        }

    def _get_confidence_level(self, score: float) -> str:
        """Devuelve un nivel de confianza legible para el usuario."""
        if score > 0.75:
            return "Alta confianza"
        elif score > 0.5:
            return "Media confianza"
        elif score > 0.3:
            return "Baja confianza"
        else:
            return "Muy baja confianza"

# Inicializar el motor QA
qa_engine = QAEngine()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó ningún archivo"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400
    
    if file and any(file.filename.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Recargar documentos
        success = qa_engine.document_handler.load_documents()
        return jsonify({
            "message": "Archivo subido correctamente",
            "reloaded": success,
            "documents": [doc['file_name'] for doc in qa_engine.document_handler.documents]
        }), 200
    
    return jsonify({"error": "Tipo de archivo no permitido"}), 400

@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({
        "documents": [doc['file_name'] for doc in qa_engine.document_handler.documents]
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "pregunta" not in data:
            return jsonify({"error": "Se requiere un campo 'pregunta'"}), 400

        question = data["pregunta"].strip()
        if not question:
            return jsonify({"error": "La pregunta no puede estar vacía"}), 400

        response = qa_engine.answer_question(question)
        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": f"Error procesando solicitud: {str(e)}"
        }), 500

if __name__ == "__main__":
    import torch
    
    # Cargar documentos al iniciar
    if not qa_engine.document_handler.load_documents():
        print("Advertencia: No se encontraron documentos para cargar")
    
    app.run(
        host=Config.HOST, 
        port=Config.PORT, 
        debug=True
    )