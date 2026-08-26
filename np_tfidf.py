import numpy as np
from glob import glob
import string
from collections import Counter, defaultdict
import math

def readFile(filepath:str) -> str:
    """
    Returns the contents of a file in a string, given its path
    """
    filecontent = ""
    with open(filepath, "r") as fptr:
        filecontent = fptr.read()
    return filecontent

def tokenize(doc:str) -> list[str]:
    """
    Returns a list of words given a string with file contents
    """
    translator = str.maketrans("", "", string.punctuation)
    return doc.strip().lower().translate(translator).split()

def buildVocab(docs:list[list[str]]) -> list[str]:
    """
    Given a list of list of words, construct vocabulary
    """
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    return list(vocab)

def countWords(doc : list[str], voc2int: dict[str, int]) -> np.ndarray:
    """
    Given a list of words in a document and a vocabulary, return count vector
    """
    n = len(voc2int)
    counts = np.zeros(n)
    idxs = [voc2int[w] for w in doc if w in voc2int]
    np.add.at(counts, idxs, 1)
    #print(counts)
    return counts

def countMatrix(tokens: list[list[str]], voc2int: dict[str, int]) -> np.ndarray:
    """
    Return 2D count matrix: word counts per document per word
    """
    ndocs = len(tokens)
    nvocab = len(voc2int)
    countMat = np.ndarray((ndocs, nvocab))
    for i, doc in enumerate(tokens):
        countMat[i] = countWords(doc, voc2int)
    return countMat


def tf(freqs : np.ndarray) -> np.ndarray:
    """
    Given a count vector for a document, return its term-frequency
    """
    return np.where(freqs > 0, 1 + np.log10(freqs, where=freqs>0), 0)

def idf(counts : np.ndarray, ndocs: int) -> np.ndarray:
    """
    Given a matrix with each row denoting a document, and each column a word, get the inverse-frequency counts
    """
    df = np.count_nonzero(counts > 0, axis=0)
    return np.log10(ndocs / df)

def tfidf(tfarr : np.ndarray, idfarr : np.ndarray) -> np.ndarray:
    tfidfarr = np.zeros(shape=tfarr.shape)
    tfidfarr = tfarr* idfarr
    return tfidfarr

def cosineSim(vec1: np.ndarray, vec2: np.ndarray) -> int:
    dot = vec1.dot(vec2)
    norm1 = np.sqrt(vec1.dot(vec1))
    norm2 = np.sqrt(vec2.dot(vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def queryVector(query: str, voc2int: dict[str, int], idfarr: np.ndarray) -> np.ndarray:
    qtokens = tokenize(query)
    qcounts = countWords(qtokens, voc2int)
    qtf = tf(qcounts)
    return tfidf(qtf, idfarr)

def search(query : str, voc2int: dict[str, int], idfarr: np.ndarray, tfarr: np.ndarray, topk: int=3) -> list[tuple[int, int]]:
    qvec = queryVector(query, voc2int, idfarr)
    scores = [(i, cosineSim(qvec, tfarr[i])) for i in range(len(tfarr))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

if __name__ == "__main__":
    #tokens of all documents
    tokens = []
    files = glob("./corpus/*")
    for f in files:
        fcontent = readFile(f)
        tokens.append(tokenize(fcontent))
    ndocs = len(tokens)
    #vocabulary
    vocab = buildVocab(tokens)
    nvocab = len(vocab)
    #word to int mapping
    voc2int = {word : idx for idx, word in enumerate(vocab)}
    freqs = countMatrix(tokens, voc2int)
    tfarr = tf(freqs)
    idfarr = idf(freqs, ndocs)
    tfarr = tfidf(tfarr, idfarr)
    query = input("Please enter your query: ")
    ranked = search(query, voc2int, idfarr, tfarr)
    for i,j in ranked:
        print(f"File {files[i]}, Score {j}")
    print(ranked)
