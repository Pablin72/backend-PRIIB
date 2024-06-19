import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def bagofwords_stemmed(directory='final/Stemmed'):
    ensure_directory_exists('final/BOW')

    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_bow_stemmed = CountVectorizer()

    X_bow_stemmed = vectorizer_bow_stemmed.fit_transform(all_sentences)

    # Guardar el vectorizador y los datos transformados.
    with open('final/BOW/Stemmed_vectorizer_bow.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_bow_stemmed, vec_file)
    
    with open('final/BOW/Stemmed_X_bow.pkl', 'wb') as xb_file:
        pickle.dump(X_bow_stemmed, xb_file)

    with open('final/BOW/Stemmed_filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # conjunto de tokens que corresponden a cada columna
    terms_bow_stemmed = vectorizer_bow_stemmed.get_feature_names_out()

    # convertir matriz dispersa en matriz densa
    # cada fila corresponde a una oración y cada columna corresponde a un token
    X_bow_stemmed = X_bow_stemmed.toarray()

    print('Bag of words  Stemmed matrix finished!')
    
    return X_bow_stemmed

def bagofwords_lemmatized(directory='final/lemmatized'):
    ensure_directory_exists('final/BOW')
    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_bow_lemmatized = CountVectorizer()

    X_bow_lemmatized = vectorizer_bow_lemmatized.fit_transform(all_sentences)

    # Guardar el vectorizador y los datos transformados.
    with open('final/BOW/lemmatized_vectorizer_bow.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_bow_lemmatized, vec_file)
    
    with open('final/BOW/lemmatized_X_bow.pkl', 'wb') as xb_file:
        pickle.dump(X_bow_lemmatized, xb_file)

    with open('final/BOW/lemmatized_filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # conjunto de tokens que corresponden a cada columna
    terms_bow_stemmed = vectorizer_bow_lemmatized.get_feature_names_out()

    # convertir matriz dispersa en matriz densa
    # cada fila corresponde a una oración y cada columna corresponde a un token
    X_bow_lemmatized = X_bow_lemmatized.toarray()

    print('Bag of words lemmatized matrix finished!')
    
    return X_bow_lemmatized

if __name__ == '__main__':
    bagofwords_stemmed()
    bagofwords_lemmatized()