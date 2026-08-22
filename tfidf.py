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

def countWords(doc : list[str], vocab: list[str]) -> list[int]:
    """
    Given a list of words in a document and a vocabulary, return count vector
    """
    n = len(vocab)
    counts = defaultdict(int)
    for word in doc:
        counts[word] += 1
    #print(counts)
    countArr = [0] * n
    for i in range(n):
        countArr[i] = counts.get(vocab[i], 0)
    return countArr

def tf(countsArr : list[int]) -> list[int]:
    """
    Given a count vector for a document, return its term-frequency
    """
    counts = countsArr.copy()
    n = len(counts)
    for i in range(n):
        if counts[i]>0:
            counts[i] = 1+math.log10(counts[i])
    return counts

def idf(counts : list[list[int]]) -> list[int]:
    """
    Given a matrix with each row denoting a document, and each column a word, get the inverse-frequency counts
    """
    ndocs = len(counts)
    nvocab = len(counts[0])
    idfarr = [0] * nvocab
    for i in range(nvocab):
        for j in range(ndocs):
            if(counts[j][i]>0):
                idfarr[i] += 1
                break
    for i in range(nvocab):
        if idfarr[i]:
            idfarr[i] = math.log10(ndocs/idfarr[i])
    return idfarr

def tfidf(tfarr : list[list[int]], idfarr : list[int]) -> list[list[int]]:
    ndoc = len(tfarr)
    nvoc = len(idfarr)
    for i in range(ndoc):
        for j in range(nvoc):
            tfarr[i][j] = tfarr[i][j] * idfarr[j]
    return tfarr

def cosineSim(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def queryVector(query, vocab, idfarr):
    qtokens = tokenize(query)
    qcounts = countWords(qtokens, vocab)
    qtf = tf(qcounts)
    return [qtf[i] * idfarr[i] for i in range(len(vocab))]

def search(query, vocab, idfarr, tfarr, topk=3):
    qvec = queryVector(query, vocab, idfarr)
    scores = [(i, cosineSim(qvec, tfarr[i])) for i in range(len(tfarr))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

if __name__ == "__main__":
    #contents of all the files
    filecontents = []
    #tokens of all documents
    tokens = []
    for f in glob("./corpus/*"):
        fcontent = readFile(f)
        tokens.append(tokenize(fcontent))
        filecontents.append(fcontent)
    ndocs = len(filecontents)
    #vocabulary
    vocab = buildVocab(tokens)
    nvocab = len(vocab)
    #frequency per document, per word
    freqs = []
    for doc in tokens:
        counts = countWords(doc, vocab)
        freqs.append(counts)

    tfarr = []
    for i in range(ndocs):
        tfarr.append(tf(freqs[i]))
    idfarr = idf(freqs)
    tfarr = tfidf(tfarr, idfarr)
    query = input("Please enter your query: ")
    ranked = search(query, vocab, idfarr, tfarr)
    print(ranked)
