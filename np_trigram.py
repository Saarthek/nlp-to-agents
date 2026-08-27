from glob import glob
import string
import random
import numpy as np

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
    ans = doc.strip().lower().translate(translator).split()
    ans.insert(0, '<s>')
    ans.insert(0, '<s>')
    ans.append('</s>')
    return ans

def buildVocab(docs:list[list[str]]) -> list[str]:
    """
    Given a list of list of words, construct vocabulary
    """
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    return sorted(list(vocab))


def returnTrigramCounts(corpus: list[list[str]], voc2int) -> tuple[np.ndarray, np.ndarray]:
    nvocab = len(voc2int)
    #counts = [[0]*nvocab]*nvocab -> same list reference for each row
    counts = np.zeros((nvocab, nvocab, nvocab))
    bigrCounts = np.zeros((nvocab, nvocab))
    trgidxs = [(voc2int[w1], voc2int[w2], voc2int[w3]) 
               for text in corpus
               for w1, w2, w3 in zip(text, text[1:], text[2:]) ]
    bigridxs = [(voc2int[w1], voc2int[w2]) 
                   for text in corpus
                   for w1, w2 in zip(text, text[1:]) ]
    row, col, depth = zip(*trgidxs)
    row2, col2 = zip(*bigridxs)
    np.add.at(counts, (row, col, depth), 1)
    np.add.at(bigrCounts, (row2, col2), 1) 
    return counts, bigrCounts


def laplaceSmoothing(trigramCounts: np.ndarray, bigramCounts: np.ndarray, voc2int: dict[str, int]) -> np.ndarray:
    nvocab = len(voc2int)
    start_idx = voc2int['<s>']
    bigr2 = bigramCounts.reshape((nvocab, nvocab, 1))
    ans = (trigramCounts+1)/(bigr2+nvocab)
    ans[:,:,start_idx] = 0
    return ans

def sampleNext(word1: str, word2: str, smoothed: np.ndarray, voc2int: dict[str, int], vocab: list[str]) -> str:
    i, j = voc2int[word1], voc2int[word2]
    probs = smoothed[i][j]
    idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
    return vocab[idx]

def generate(smoothed: np.ndarray, voc2int: dict[str, int], vocab: list[str], max_len: int=20) -> list[str]:
    words = ['<s>', '<s>']
    for _ in range(max_len):
        next_word = sampleNext(words[-2], words[-1], smoothed, voc2int, vocab)
        if next_word == '</s>':
            break
        words.append(next_word)
    return words[2:]

if __name__ == "__main__":
    #tokens of all documents
    tokens = []
    files = glob("./corpus/*")
    for f in files:
        fcontent = readFile(f)
        tokens.append(tokenize(fcontent))
    ndocs = len(tokens)
    vocab = buildVocab(tokens)
    voc2int = {word:idx for idx,word in enumerate(vocab)}
    trigramCounts, bigramCounts = returnTrigramCounts(tokens, voc2int)
    smoothed = laplaceSmoothing(trigramCounts, bigramCounts, voc2int)
    ans = generate(smoothed, voc2int, vocab)
    ans = ' '.join(ans)
    print(ans)