from flask import Flask, request, jsonify

app = Flask(__name__)

# Inicialización global
documents = ["Texto 1 extraído", "Texto 2 extraído"]  # Sustituir con el texto real de los PDFs
index, vectorizer = create_faiss_index(documents)
pages = documents

@app.route("/chat", methods=["POST"])
def chat():
    try:
        pregunta = request.json.get("pregunta")
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        contextos = buscar_contextos_relevantes(pregunta, pages, index, vectorizer)
        respuesta = responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']})
    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)