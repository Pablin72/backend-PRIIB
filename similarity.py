import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from scipy.sparse import csr_matrix

def lemmatized_tfidf_cosine_similarity_search(query):
    # cargar la data transformada y los vectores
    with open('final/TF-IDF/lemmatized_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/lemmatized_X_tfidf.pkl', 'rb') as xt_file:
        X_tfidf = pickle.load(xt_file)
    
    with open('final/TF-IDF/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_tfidf.transform([query])

    # calcular la similitud coseno
    cosine_sim_scores = cosine_similarity(X_tfidf, query_vector)

    # se crea un dataframe
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

    # se organiza el documento
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def stemmed_tfidf_cosine_similarity_search(query):
    # cargar la data transformada y los vectores
    with open('final/TF-IDF/Stemmed_vectorizer_tfidf.pkl', 'rb') as vec_file:
        vectorizer_tfidf = pickle.load(vec_file)
    
    with open('final/TF-IDF/Stemmed_X_tfidf.pkl', 'rb') as xt_file:
        X_tfidf = pickle.load(xt_file)
    
    with open('final/TF-IDF/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    query_vector = vectorizer_tfidf.transform([query])

    # calcular la similitud coseno
    cosine_sim_scores = cosine_similarity(X_tfidf, query_vector)

     # se crea un dataframe
    similarity_df = pd.DataFrame({'Filename': filenames, 'Cosine_Similarity': cosine_sim_scores.flatten()})

     # se organiza el documento
    similarity_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)

    return similarity_df

def jaccard_similarity(query_vector, corpus_matrix):

    #Se asegura de que ambas matrices tengan el mismo número de columnas.
    if query_vector.shape[1] != corpus_matrix.shape[1]:
        raise ValueError("Dimension mismatch between query_vector and corpus_matrix")

    num_docs = corpus_matrix.shape[0]
    jaccard_scores = []

    for i in range(num_docs):
        corpus_row = corpus_matrix.getrow(i)

        try:
            # Mínimo y máximo en cuanto al elemento
            intersection = query_vector.minimum(corpus_row)
            union = query_vector.maximum(corpus_row)

            # Suma para obtener los recuentos de intersecciones y uniones del documento.
            intersection_sum = intersection.sum()
            union_sum = union.sum()

            jaccard_score = intersection_sum / union_sum if union_sum != 0 else 0
            jaccard_scores.append(jaccard_score)
        except Exception as e:
            print(f"Error during intersection or union calculation for document {i}:", e)
            jaccard_scores.append(0)

    return np.array(jaccard_scores)

def stemmed_bow_jaccard_similarity_search(query):
    # cargar la data transformada y los vectores
    with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/Stemmed_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # convertir la matriz de BoW a binaria
    X_bow_binary = (X_bow > 0).astype(int)

    # Transformar la consulta a su representación binaria BoW
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

    # Se asegura de que ambas matrices sean matrices de CRS
    query_vector_binary = csr_matrix(query_vector_binary)
    X_bow_binary = csr_matrix(X_bow_binary)

    # Calcular puntuaciones de similitud de Jaccard
    jaccard_sim_scores = jaccard_similarity(query_vector_binary, X_bow_binary)

    if jaccard_sim_scores is None:
        print("Error in calculating Jaccard similarity scores")
        return None

    # Verifique que las longitudes coincidan
    if len(filenames) != len(jaccard_sim_scores):
        print(f"Length mismatch: {len(filenames)} filenames vs {len(jaccard_sim_scores)} scores")
        return None

    # Se crea un dataframe
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Ordenar documentos según puntuaciones de similitud
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df

def lemmatized_bow_jaccard_similarity_search(query):
    # cargar la data transformada y los vectores
    with open('final/BOW/lemmatized_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    
    with open('final/BOW/lemmatized_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    
    with open('final/BOW/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)

    # convertir la matriz de BoW a binaria
    X_bow_binary = (X_bow > 0).astype(int)

    # Transformar la consulta a su representación binaria BoW
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)

   # Se asegura de que ambas matrices sean matrices de CRS
    query_vector_binary = csr_matrix(query_vector_binary)
    X_bow_binary = csr_matrix(X_bow_binary)

    # Calcular puntuaciones de similitud de Jaccard
    jaccard_sim_scores = jaccard_similarity(query_vector_binary, X_bow_binary)

    if jaccard_sim_scores is None:
        print("Error in calculating Jaccard similarity scores")
        return None

    # Verifique que las longitudes coincidan
    if len(filenames) != len(jaccard_sim_scores):
        print(f"Length mismatch: {len(filenames)} filenames vs {len(jaccard_sim_scores)} scores")
        return None

    # Se crea un dataframe
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})

    # Ordenar documentos según puntuaciones de similitud
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)

    return similarity_df