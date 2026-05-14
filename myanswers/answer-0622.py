from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_sentencias(df):
    textos = df["texto"]

    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(textos).toarray()
    palabras = list(vectorizer.get_feature_names_out())

    return matriz_tfidf, palabras