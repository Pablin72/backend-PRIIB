import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from scipy.sparse import csr_matrix

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


# def jaccard_similarity(query_vector, corpus_matrix):

#     query_vector = csr_matrix(query_vector)
#     print("query_vector", query_vector)
#     intersection = query_vector.minimum(corpus_matrix).sum(axis=1)
#     print("intersection")
#     union = query_vector.maximum(corpus_matrix).sum(axis=1)
#     print("union")
#     jaccard_scores = intersection / union
#     print("jaccard_scores and returning")
    
#     return jaccard_scores.A1

def jaccard_similarity(query_vector, corpus_matrix):
    print("query_vector shape:", query_vector.shape)
    print("corpus_matrix shape:", corpus_matrix.shape)

    # Ensure both matrices have the same number of columns
    if query_vector.shape[1] != corpus_matrix.shape[1]:
        raise ValueError("Dimension mismatch between query_vector and corpus_matrix")

    num_docs = corpus_matrix.shape[0]
    jaccard_scores = []

    for i in range(num_docs):
        corpus_row = corpus_matrix.getrow(i)

        try:
            # Element-wise minimum and maximum
            intersection = query_vector.minimum(corpus_row)
            union = query_vector.maximum(corpus_row)

            print(f"Document {i} intersection values:", intersection.toarray())  # Convert to dense array for printing
            print(f"Document {i} union values:", union.toarray())  # Convert to dense array for printing

            # Sum to get the intersection and union counts for the document
            intersection_sum = intersection.sum()
            union_sum = union.sum()

            print(f"Document {i} intersection sum:", intersection_sum)
            print(f"Document {i} union sum:", union_sum)

            jaccard_score = intersection_sum / union_sum if union_sum != 0 else 0
            jaccard_scores.append(jaccard_score)
        except Exception as e:
            print(f"Error during intersection or union calculation for document {i}:", e)
            jaccard_scores.append(0)

    return np.array(jaccard_scores)

# def stemmed_bow_jaccard_similarity_search(query):
#     # Load the vectorizer and transformed data
#     with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'rb') as vec_file:
#         vectorizer_bow = pickle.load(vec_file)
#     print("open vectorizer")
    
#     with open('final/BOW/Stemmed_X_bow.pkl', 'rb') as xb_file:
#         X_bow = pickle.load(xb_file)
#     print("open X_bow")
    
#     with open('final/BOW/Stemmed_filenames.pkl', 'rb') as f_file:
#         filenames = pickle.load(f_file)
#     print("open filenames")

#     # Convert the BoW matrix to binary
#     X_bow_binary = (X_bow > 0).astype(int)
#     print("X_bow_binary to binary")

#     # Transform the query to its binary BoW representation
#     query_vector = vectorizer_bow.transform([query])
#     query_vector_binary = (query_vector > 0).astype(int)
#     print("query_vector_binary to binary")

#     # Calculate Jaccard similarity scores
#     jaccard_sim_scores = jaccard_similarity(query_vector_binary,X_bow_binary)
#     print("jaccard_sim_scores")

#     # Create a DataFrame to store document filenames and their similarity scores
#     similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})
#     print("similarity_df")

#     # Sort documents based on similarity scores
#     similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)
#     print("sort documents")

#     return similarity_df

def stemmed_bow_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    print("Loaded vectorizer")
    
    with open('final/BOW/Stemmed_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    print("Loaded X_bow")
    
    with open('final/BOW/Stemmed_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)
    print("Loaded filenames")

    # Convert the BoW matrix to binary
    X_bow_binary = (X_bow > 0).astype(int)
    print("Converted X_bow to binary")

    # Transform the query to its binary BoW representation
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)
    print("Converted query vector to binary")

    # Ensure both matrices are CSR matrices
    query_vector_binary = csr_matrix(query_vector_binary)
    X_bow_binary = csr_matrix(X_bow_binary)

    # Debug shapes and types
    print("query_vector_binary type:", type(query_vector_binary), "shape:", query_vector_binary.shape)
    print("query_vector_binary values:", query_vector_binary.toarray())
    print("X_bow_binary type:", type(X_bow_binary), "shape:", X_bow_binary.shape)
    print("X_bow_binary values:", X_bow_binary.toarray())

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(query_vector_binary, X_bow_binary)
    print("Calculated Jaccard similarity scores")

    if jaccard_sim_scores is None:
        print("Error in calculating Jaccard similarity scores")
        return None

    # Verify the lengths match
    if len(filenames) != len(jaccard_sim_scores):
        print(f"Length mismatch: {len(filenames)} filenames vs {len(jaccard_sim_scores)} scores")
        return None

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})
    print("Created similarity DataFrame")

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)
    print("Sorted documents by similarity scores")

    return similarity_df

def lemmatized_bow_jaccard_similarity_search(query):
    # Load the vectorizer and transformed data
    with open('final/BOW/lemmatized_vectorizer_bow.pkl', 'rb') as vec_file:
        vectorizer_bow = pickle.load(vec_file)
    print("Loaded vectorizer")
    
    with open('final/BOW/lemmatized_X_bow.pkl', 'rb') as xb_file:
        X_bow = pickle.load(xb_file)
    print("Loaded X_bow")
    
    with open('final/BOW/lemmatized_filenames.pkl', 'rb') as f_file:
        filenames = pickle.load(f_file)
    print("Loaded filenames")

    # Convert the BoW matrix to binary
    X_bow_binary = (X_bow > 0).astype(int)
    print("Converted X_bow to binary")

    # Transform the query to its binary BoW representation
    query_vector = vectorizer_bow.transform([query])
    query_vector_binary = (query_vector > 0).astype(int)
    print("Converted query vector to binary")

    # Ensure both matrices are CSR matrices
    query_vector_binary = csr_matrix(query_vector_binary)
    X_bow_binary = csr_matrix(X_bow_binary)

    # Debug shapes and types
    print("query_vector_binary type:", type(query_vector_binary), "shape:", query_vector_binary.shape)
    print("query_vector_binary values:", query_vector_binary.toarray())
    print("X_bow_binary type:", type(X_bow_binary), "shape:", X_bow_binary.shape)
    print("X_bow_binary values:", X_bow_binary.toarray())

    # Calculate Jaccard similarity scores
    jaccard_sim_scores = jaccard_similarity(query_vector_binary, X_bow_binary)
    print("Calculated Jaccard similarity scores")

    if jaccard_sim_scores is None:
        print("Error in calculating Jaccard similarity scores")
        return None

    # Verify the lengths match
    if len(filenames) != len(jaccard_sim_scores):
        print(f"Length mismatch: {len(filenames)} filenames vs {len(jaccard_sim_scores)} scores")
        return None

    # Create a DataFrame to store document filenames and their similarity scores
    similarity_df = pd.DataFrame({'Filename': filenames, 'Jaccard_Similarity': jaccard_sim_scores})
    print("Created similarity DataFrame")

    # Sort documents based on similarity scores
    similarity_df = similarity_df.sort_values(by='Jaccard_Similarity', ascending=False)
    print("Sorted documents by similarity scores")

    return similarity_df