import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

def lemmatized_bow_cosine_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/lemmatized_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/lemmatized_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_bow.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_bow, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def stemmed_bow_cosine_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/Stemmed_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_bow.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_bow, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def lemmatized_tfidf_cosine_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/TF-IDF/lemmatized_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/lemmatized_X_tfidf.pkl', 'rb') as xt_file:
        X_tfidf = pickle.load(xt_file)
    
    with open('final/TF-IDF/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_tfidf.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_tfidf, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def stemmed_tfidf_cosine_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/TF-IDF/Stemmed_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/Stemmed_X_tfidf.pkl', 'rb') as xt_file:
        X_tfidf = pickle.load(xt_file)
    
    with open('final/TF-IDF/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_tfidf.transform([query])

    # Calculate cosine similarity scores
    cosine_sim_scores = cosine_similarity(X_tfidf, query_vector)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def jaccard_similarity(X, Y):
    """Calculate the Jaccard similarity between two binary matrices."""
    intersection = np.logical_and(X, Y).sum(axis=1)
    union = np.logical_or(X, Y).sum(axis=1)
    return intersection / union

def stemmed_bow_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/Stemmed_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # Convert the BoW matrix to binary
    X_bow_binary = (X_bow > 0).astype(int)

    # Transform the query to its binary BoW representation
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(X_bow_binary, query_vector_binary)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df

def lemmatized_bow_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/lemmatized_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/lemmatized_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # Convert the BoW matrix to binary
    X_bow_binary = (X_bow > 0).astype(int)

    # Transform the query to its binary BoW representation
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(X_bow_binary, query_vector_binary)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df

def stemmed_tfidf_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/TF-IDF/Stemmed_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/Stemmed_X_tfidf.pkl', 'rb') as xb_file:
        X_tfidf = pickle.load(xb_file)
    
    with open('final/TF-IDF/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # Convert the BoW matrix to binary
    X_tfidf_binary = (X_tfidf > 0).astype(int)

    # Transform the query to its binary TF-IDF representation
    query_vector = vectorizer_tfidf.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(X_tfidf_binary, query_vector_binary)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df

def lemmatized_tfidf_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/TF-IDF/lemmatized_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/lemmatized_X_tfidf.pkl', 'rb') as xb_file:
        X_tfidf = pickle.load(xb_file)
    
    with open('final/TF-IDF/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # Convert the BoW matrix to binary
    X_tfidf_binary = (X_tfidf > 0).astype(int)

    # Transform the query to its binary TF-IDF representation
    query_vector = vectorizer_tfidf.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(X_tfidf_binary, query_vector_binary)

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df