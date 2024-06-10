from flask import Flask, request, jsonify
from flask_cors import CORS
from similarity import cosine_similarity_search
import pandas as pd  # Importa pandas

import bagofwords as bow


app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def serve_index():
    return "This is the main page"

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    directory = data.get('directory', 'reuters/final_txt')
    query = data.get('query', '')

    similarity_df = cosine_similarity_search('reuters/final_txt', query)

    # Convertir DataFrame a diccionario antes de devolverlo como JSON
    similarity_dict = similarity_df.to_dict(orient='records')

    return jsonify(similarity_dict)

@app.route('/cosine', methods=['POST'])
def cosine():
    data = request.json
    query = data.get('query')

    similarity_df = cosine_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

# @app.route('/jaccard', methods=['POST'])
# def jaccard():
#     data = request.json
#     query = data.get('query')

#     similarity_df = jaccard_similarity_search(query)
#     similarity_json = similarity_df.to_dict(orient='records')

#     return jsonify(similarity_json)

# @app.route('/tfidf', methods=['POST'])
# def tfidf():
#     data = request.json
#     query = data.get('query')

#     similarity_df = bow.tfidf_similarity_search(query)
#     similarity_json = similarity_df.to_dict(orient='records')

#     return jsonify(similarity_json)

if __name__ == '__main__':
    app.run(debug=True)
