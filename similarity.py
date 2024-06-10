import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def cosine_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('reuters/vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('reuters/X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('reuters/filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_bow.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_bow, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

# def jaccard_similarity_search(query):
