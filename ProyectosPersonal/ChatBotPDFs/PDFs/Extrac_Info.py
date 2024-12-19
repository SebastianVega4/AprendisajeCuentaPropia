import pdfplumber

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text.strip() if text else "El PDF no contiene texto legible."
    except Exception as e:
        return f"Error al procesar el PDF: {e}"

def extract_tables_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            tables = []
            for page in pdf.pages:
                tables += page.extract_tables()
            return tables
    except Exception as e:
        return f"Error al procesar las tablas: {e}"

def create_txt_with_text(text, name_txt):
    try:
        with open(f"{name_txt}.txt", 'w', encoding="UTF-8") as archivo:
            archivo.write(text)
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
