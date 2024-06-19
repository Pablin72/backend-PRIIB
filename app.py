from flask import Flask, request, jsonify
from flask_cors import CORS
import similarity
import re
import os
import spacy
import snowballstemmer

DOCUMENTS_DIR = 'reuters/training_txt'

app = Flask(__name__)
CORS(app)

# Función para lematizar y stematizar texto
def preprocess_text(text):
    doc = nlp(text)
    stemmed_tokens = [stemmer.stemWord(token.text) for token in doc]
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(stemmed_tokens), ' '.join(lemmatized_tokens)

def clean_query(query):
    # Limpiamos la query
    stop_words_file = 'reuters/stopwords.txt'
    with open(stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().split())

    query = query.lower()
    cleaned_query = ' '.join([word for word in query.split() if word.lower() not in stop_words])
    cleaned_query = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_query)
    stemmed_query, lemmatized_query = preprocess_text(cleaned_query)
    return lemmatized_query

@app.route('/')
def serve_index():
    return "This is the main page"

@app.route('/doc/', methods=['GET'])
def get_doc():
    doc_id = request.args.get('doc_id')
    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400
    
    file_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}")  # asumimos que los documentos tienen la extension .txt
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r') as file:
        content = file.read()
    return {"doc_id": doc_id, "content": content}

@app.route('/lemmatized/tfidf/cosine', methods=['POST'])
def lemmatized_tfidf_cosine():
    data = request.json
    query = data.get('query')
    query = clean_query(query)

    similarity_df = similarity.lemmatized_tfidf_cosine_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

@app.route('/stemmed/tfidf/cosine', methods=['POST'])
def stemmed_tfidf_cosine():
    data = request.json
    query = data.get('query')
    query = clean_query(query)

    similarity_df = similarity.stemmed_tfidf_cosine_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

@app.route('/stemmed/bow/jaccard', methods=['POST'])
def stemmed_bow_jaccard():
    data = request.json
    query = data.get('query')
    print("my query: ", query)
    query = clean_query(query)

    similarity_df = similarity.stemmed_bow_jaccard_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')
    print("sorting results and return")

    return jsonify(similarity_json)

@app.route('/lemmatized/bow/jaccard', methods=['POST'])
def lemmatized_bow_jaccard():
    data = request.json
    query = data.get('query')
    query = clean_query(query)

    similarity_df = similarity.lemmatized_bow_jaccard_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

if __name__ == '__main__':
    # Cargar el modelo de lenguaje de spaCy
    nlp = spacy.load('en_core_web_sm')
    # Inicializar el stemmer
    stemmer = snowballstemmer.stemmer('english')
    app.run(debug=True)
