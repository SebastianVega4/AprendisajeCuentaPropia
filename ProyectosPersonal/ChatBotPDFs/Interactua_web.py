from flask import Flask, request, jsonify

from ProyectosPersonal.ChatBotPDFs.PDFs import Inde_data, Compresion_Transformers as Trasfor

app = Flask(__name__)

# Inicialización global
documents = ["Texto 1 extraído", "Texto 2 extraído"]  # Sustituir con el texto real de los PDFs
index, vectorizer = Inde_data.create_faiss_index(documents)
pages = documents

@app.route("/chat", methods=["POST"])
def chat():
    try:
        pregunta = request.json.get("pregunta")
        if not pregunta:
            return jsonify({"error": "La pregunta no puede estar vacía."}), 400

        contextos = Trasfor.buscar_contextos_relevantes(pregunta, pages, index, vectorizer)
        respuesta = Trasfor.responder_pregunta(pregunta, contextos)
        return jsonify({"answer": respuesta['answer'], "score": respuesta['score']})
    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)