import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

def create_faiss_index(documents):
    """Crea un índice FAISS a partir de los documentos."""
    try:
        vectorizer = TfidfVectorizer(stop_words="spanish")
        tfidf_matrix = vectorizer.fit_transform([doc['content'] for doc in documents]).toarray()
        dim = tfidf_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(tfidf_matrix.astype("float32"))
        return index, vectorizer
    except Exception as e:
        print(f"Error al crear el índice FAISS: {e}")
        return None, None