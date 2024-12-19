import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un archivo PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return ""

def load_pdfs_from_directory(directory):
    """Carga todos los PDFs de un directorio y extrae su texto."""
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            full_path = os.path.join(directory, file_name)
            text = extract_text_from_pdf(full_path)
            if text:
                documents.append({"file_name": file_name, "content": text})
    return documents