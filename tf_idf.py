import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def tfidf_lemmatized(directory='final/lemmatized'):

    ensure_directory_exists('final/TF-IDF')

    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_tfidf_lemmatized = TfidfVectorizer()

    X_tfidf_lemmatized = vectorizer_tfidf_lemmatized.fit_transform(all_sentences)

    # Guardar el vectorizador y los datos transformados.
    with open('final/TF-IDF/lemmatized_vectorizer_tfidf.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_tfidf_lemmatized, vec_file)
    
    with open('final/TF-IDF/lemmatized_X_tfidf.pkl', 'wb') as xt_file:
        pickle.dump(X_tfidf_lemmatized, xt_file)

    with open('final/TF-IDF/lemmatized_filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # conjunto de tokens que corresponden a cada columna
    terms_tfidf_lemmatized = vectorizer_tfidf_lemmatized.get_feature_names_out()

    # convertir matriz dispersa en matriz densa
    # cada fila corresponde a una oración y cada columna corresponde a un token
    X_tfidf_lemmatized = X_tfidf_lemmatized.toarray()

    print('TF-IDF lemmatized matrix finished!')
    
    return X_tfidf_lemmatized

def tfidf_stemmed(directory='final/stemmed'):

    ensure_directory_exists('final/TF-IDF')

    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_tfidf_stemmed = TfidfVectorizer()

    X_tfidf_stemmed = vectorizer_tfidf_stemmed.fit_transform(all_sentences)

    # Guardar el vectorizador y los datos transformados.
    with open('final/TF-IDF/stemmed_vectorizer_tfidf.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_tfidf_stemmed, vec_file)
    
    with open('final/TF-IDF/stemmed_X_tfidf.pkl', 'wb') as xt_file:
        pickle.dump(X_tfidf_stemmed, xt_file)

    with open('final/TF-IDF/stemmed_filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # conjunto de tokens que corresponden a cada columna
    terms_tfidf = vectorizer_tfidf_stemmed.get_feature_names_out()

    # convertir matriz dispersa en matriz densa
    # cada fila corresponde a una oración y cada columna corresponde a un token
    X_tfidf_stemmed = X_tfidf_stemmed.toarray()

    print('TF-IDF stemmed matrix finished!')
    
    return X_tfidf_stemmed

if __name__ == '__main__':
    tfidf_lemmatized()
    tfidf_stemmed()
