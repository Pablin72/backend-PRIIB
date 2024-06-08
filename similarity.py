import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_search(directory, query):
    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(all_sentences)
    query_vector = vectorizer_bow.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_bow, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df
