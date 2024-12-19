from flask import Flask, request, jsonify

app = Flask(__name__)

# Variables globales
documents = []  # Lista para almacenar documentos
index, vectorizer = None, None  # Inicialización de FAISS

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Endpoint para cargar PDFs y actualizar el índice."""
    try:
        file = request.files['file']
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Solo se permiten archivos PDF"}), 400

        text = extract_text_from_pdf(file)
        if text:
            documents.append({"file_name": file.filename, "content": text})
            global index, vectorizer
            index, vectorizer = create_faiss_index(documents)
            return jsonify({"message": "Archivo procesado correctamente"}), 200
        else:
            return jsonify({"error": "El archivo no contiene texto legible"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al cargar el archivo: {e}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint para responder preguntas."""
    try:
        pregunta = request.json.get("pregunta")
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        if not index or not vectorizer:
            return jsonify({"error": "No hay datos indexados"}), 400

        contextos = buscar_contextos_relevantes(pregunta, [doc['content'] for doc in documents], index, vectorizer)
        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']}), 200
    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {e}"}), 500