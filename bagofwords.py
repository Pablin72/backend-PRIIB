import os
from sklearn.feature_extraction.text import CountVectorizer

def bagofwords(directory='reuters/final_txt'):
    filenames = []
    all_sentences = []

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            all_sentences.append(content)
            filenames.append(filename)

    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(all_sentences)

    terms_bow = vectorizer_bow.get_feature_names_out()

    X_bow = X_bow.toarray()
    
    return X_bow
