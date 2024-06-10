import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def bagofwords(directory='reuters/final_txt'):
    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_bow = CountVectorizer()
    # matrix of tokens counts
    # row -> document
    # column -> token
    X_bow = vectorizer_bow.fit_transform(all_sentences)

    # Save the vectorizer and transformed data
    with open('reuters/vectorizer_bow.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer_bow, vec_file)
    
    with open('reuters/X_bow.pkl', 'wb') as xb_file:
        pickle.dump(X_bow, xb_file)

    with open('reuters/filenames.pkl', 'wb') as f_file:
        pickle.dump(filenames, f_file)

    # array of tokens that correspond to each column
    terms_bow = vectorizer_bow.get_feature_names_out()

    # convert sparse matrix to dense matrix
    # each row corresponds to a sentence and each column corresponds to a token
    X_bow = X_bow.toarray()

    print('Bag of words matrix finished!')
    
    return X_bow

if __name__ == '__main__':
    bagofwords()