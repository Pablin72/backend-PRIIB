from flask import Flask, request, jsonify
from flask_cors import CORS
import similarity

# app = Flask(__name__, static_folder='static')
app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return "This is the main page"

@app.route('/lemmatized/tfidf/cosine', methods=['POST'])
def lemmatized_tfidf_cosine():
    data = request.json
    query = data.get('query')

    similarity_df = similarity.lemmatized_tfidf_cosine_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

@app.route('/stemmed/tfidf/cosine', methods=['POST'])
def stemmed_tfidf_cosine():
    data = request.json
    query = data.get('query')

    similarity_df = similarity.stemmed_tfidf_cosine_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

@app.route('/stemmed/bow/jaccard', methods=['POST'])
def stemmed_bow_jaccard():
    data = request.json
    query = data.get('query')

    similarity_df = similarity.stemmed_bow_jaccard_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

@app.route('/lemmatized/bow/jaccard', methods=['POST'])
def lemmatized_bow_jaccard():
    data = request.json
    query = data.get('query')

    similarity_df = similarity.lemmatized_bow_jaccard_similarity_search(query)
    print(similarity_df)
    similarity_json = similarity_df.to_dict(orient='records')

    return jsonify(similarity_json)

if __name__ == '__main__':
    app.run(debug=True)
