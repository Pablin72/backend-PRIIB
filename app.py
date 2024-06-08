from flask import Flask, request, jsonify
from similarity import cosine_similarity_search
import pandas as pd  # Importa pandas


app = Flask(__name__, static_folder='static')

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

if __name__ == '__main__':
    app.run(debug=True)
