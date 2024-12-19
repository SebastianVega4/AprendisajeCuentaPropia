import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def create_faiss_index(documents):
    try:
        vectorizer = TfidfVectorizer(stop_words="spanish")
        tfidf_matrix = vectorizer.fit_transform(documents).toarray()

        dim = tfidf_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(tfidf_matrix, dtype=np.float32))

        return index, vectorizer
    except Exception as e:
        print(f"Error al crear el índice FAISS: {e}")
        return None, None