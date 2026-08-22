import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob

def readFile(filepath:str) -> str:
    """
    Returns the contents of a file in a string, given its path
    """
    filecontent = ""
    with open(filepath, "r") as fptr:
        filecontent = fptr.read()
    return filecontent

if __name__ == "__main__":
    #contents of all the files
    filecontents = []
    files = glob("./corpus/*")
    for f in files:
        fcontent = readFile(f)
        filecontents.append(fcontent)
    ndocs = len(filecontents)
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(filecontents)
    query = input('Enter your query here: ')
    qvec = tfidf.transform([query])
    print('\nidf values:')
    for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(ele1, ':', ele2)
    #vocabulary
    vocab = tfidf.vocabulary_
    print(f'Vocabulary: {vocab}')
    nvocab = len(vocab)
    print('\ntf-idf value:')
    print(result.toarray())
    sims = cosine_similarity(qvec, result).flatten()

    print('\nquery similarity scores:')
    ranked = sorted(zip(files, sims), key=lambda x: x[1], reverse=True)
    for fname, score in ranked:
        print(f'{fname}: {score:.4f}')
