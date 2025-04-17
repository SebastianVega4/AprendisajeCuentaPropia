python -m spacy download es_core_news_sm


// por si no sirve todo el mpn instal -r requirements.txt

pip install nltk numpy
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
pip install flask
pip install flask-cors==3.0.10
pip install transformers
pip install scikit-learn
pip install pdfplumber
pip install python-docx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



pip install flask transformers pdfplumber python-docx pillow
python -c "import sklearn; print(sklearn.__version__)"
python -c "import flask, transformers, sklearn, torch, nltk; print('Todas las dependencias principales están instaladas')"


pip list | grep -E "flask|transformers|pdfplumber|python-docx"
pip uninstall flask transformers pdfplumber python-docx -y
pip install --upgrade flask transformers pdfplumber python-docx
pip install flask==2.3.2 transformers==4.30.2 pdfplumber==0.9.0 python-docx==0.8.11