import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def tfidf(directory='final/lemmatized'):
    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_tfidf = TfidfVectorizer()
    # matrix of tf-idf scores
    # row -> document
    # column -> token
    X_tfidf = vectorizer_tfidf.fit_transform(all_sentences)

    # Save the vectorizer and transformed data
    with open('final/TF-IDF/lemmatized_vectorizer_tfidf.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_tfidf, vec_file)
    
    with open('final/TF-IDF/lemmatized_X_tfidf.pkl', 'wb') as xt_file:
        pickle.dump(X_tfidf, xt_file)

    with open('final/TF-IDF/lemmatized_filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # array of tokens that correspond to each column
    terms_tfidf = vectorizer_tfidf.get_feature_names_out()

    # convert sparse matrix to dense matrix
    # each row corresponds to a sentence and each column corresponds to a token
    X_tfidf = X_tfidf.toarray()

    print('TF-IDF matrix finished!')
    
    return X_tfidf

if __name__ == '__main__':
    tfidf()
