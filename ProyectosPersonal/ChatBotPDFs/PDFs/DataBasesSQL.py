import sqlite3

def init_db():
    """Inicializa la base de datos."""
    conn = sqlite3.connect("documents.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(file_name, content):
    """Guarda un documento en la base de datos."""
    conn = sqlite3.connect("documents.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (file_name, content) VALUES (?, ?)", (file_name, content))
    conn.commit()
    conn.close()