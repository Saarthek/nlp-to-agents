from glob import glob
import string
import random

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
    return list(vocab)


def returnTrigramCounts(corpus: list[list[str]], voc2int) -> tuple[list[list[list[int]]], list[list[int]]]:
    nvocab = len(voc2int)
    #counts = [[0]*nvocab]*nvocab -> same list reference for each row
    counts = [[[0]*nvocab for i in range(nvocab)] for j in range(nvocab)]
    bigrCounts = [[0]*nvocab for _ in range(nvocab)]
    for text in corpus:
        for word1, word2, word3 in zip(text, text[1:], text[2:]):
            counts[voc2int[word1]][voc2int[word2]][voc2int[word3]] += 1
        for word1, word2 in zip(text, text[1:]):
            bigrCounts[voc2int[word1]][voc2int[word2]] += 1
            
    return counts, bigrCounts


def laplaceSmoothing(trigramCounts, bigramCounts, voc2int):
    nvocab = len(voc2int)
    start_idx = voc2int['<s>']
    ans = [[[0.0]*nvocab for _ in range(nvocab)] for _ in range(nvocab)]
    for i in range(nvocab):
        for j in range(nvocab):
            for k in range(nvocab):
                if k == start_idx:
                    ans[i][j][k] = 0.0
                else:
                    ans[i][j][k] = (trigramCounts[i][j][k] + 1) / (bigramCounts[i][j] + nvocab - 1)
    return ans

def sampleNext(word1, word2, smoothed, voc2int, vocab):
    i, j = voc2int[word1], voc2int[word2]
    probs = smoothed[i][j]
    idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
    return vocab[idx]

def generate(smoothed, voc2int, vocab, max_len=20):
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