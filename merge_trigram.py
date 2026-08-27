from glob import glob
import string
import random
import numpy as np

def readFile(filepath: str) -> str:
    """
    Returns the contents of a file in a string, given its path
    """
    with open(filepath, "r") as fptr:
        return fptr.read()

def tokenize(doc: str) -> list[str]:
    """
    Returns a list of words given a string with file contents
    """
    translator = str.maketrans("", "", string.punctuation)
    ans = doc.strip().lower().translate(translator).split()
    ans.insert(0, '<s>')
    ans.insert(0, '<s>')
    ans.append('</s>')
    return ans

def buildVocab(docs: list[list[str]]) -> list[str]:
    """
    Given a list of list of words, construct vocabulary
    """
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    return sorted(list(vocab))

# ============================================================
# CODE 1: LIST IMPLEMENTATION
# ============================================================

def returnTrigramCounts_list(corpus: list[list[str]], voc2int: dict[str, int]) -> tuple[list[list[list[int]]], list[list[int]]]:
    nvocab = len(voc2int)
    counts = [[[0] * nvocab for _ in range(nvocab)] for _ in range(nvocab)]
    bigrCounts = [[0] * nvocab for _ in range(nvocab)]
    for text in corpus:
        for word1, word2, word3 in zip(text, text[1:], text[2:]):
            counts[voc2int[word1]][voc2int[word2]][voc2int[word3]] += 1
        for word1, word2 in zip(text, text[1:]):
            bigrCounts[voc2int[word1]][voc2int[word2]] += 1
    return counts, bigrCounts

def laplaceSmoothing_list(
    trigramCounts: list[list[list[int]]],
    bigramCounts: list[list[int]],
    voc2int: dict[str, int]
) -> list[list[list[float]]]:
    nvocab = len(voc2int)
    start_idx = voc2int['<s>']
    ans = [[[0.0] * nvocab for _ in range(nvocab)] for _ in range(nvocab)]
    for i in range(nvocab):
        for j in range(nvocab):
            for k in range(nvocab):
                if k == start_idx:
                    ans[i][j][k] = 0.0
                else:
                    ans[i][j][k] = (
                        trigramCounts[i][j][k] + 1
                    ) / (
                        bigramCounts[i][j] + nvocab
                    )
    return ans

def generate_list(
    smoothed: list[list[list[float]]],
    voc2int: dict[str, int],
    vocab: list[str],
    max_len: int = 20
) -> list[str]:
    words = ['<s>', '<s>']
    for _ in range(max_len):
        i = voc2int[words[-2]]
        j = voc2int[words[-1]]
        probs = smoothed[i][j]
        idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
        next_word = vocab[idx]
        if next_word == '</s>':
            break
        words.append(next_word)
    return words[2:]

# ============================================================
# CODE 2: NUMPY IMPLEMENTATION
# ============================================================

def returnTrigramCounts_numpy(
    corpus: list[list[str]],
    voc2int: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    nvocab = len(voc2int)
    counts = np.zeros(
        (nvocab, nvocab, nvocab),
        dtype=np.int64
    )
    bigrCounts = np.zeros(
        (nvocab, nvocab),
        dtype=np.int64
    )
    trgidxs = [
        (voc2int[w1], voc2int[w2], voc2int[w3])
        for text in corpus
        for w1, w2, w3 in zip(text, text[1:], text[2:])
    ]
    bigridxs = [
        (voc2int[w1], voc2int[w2])
        for text in corpus
        for w1, w2 in zip(text, text[1:])
    ]
    if trgidxs:
        row, col, depth = zip(*trgidxs)
        np.add.at(counts, (row, col, depth), 1)
    if bigridxs:
        row2, col2 = zip(*bigridxs)
        np.add.at(bigrCounts, (row2, col2), 1)
    return counts, bigrCounts

def laplaceSmoothing_numpy(
    trigramCounts: np.ndarray,
    bigramCounts: np.ndarray,
    voc2int: dict[str, int]
) -> np.ndarray:
    nvocab = len(voc2int)
    start_idx = voc2int['<s>']
    bigr2 = bigramCounts.reshape((nvocab, nvocab, 1))
    ans = (trigramCounts + 1) / (bigr2 + nvocab)
    ans[:, :, start_idx] = 0.0
    return ans

def generate_numpy(
    smoothed: np.ndarray,
    voc2int: dict[str, int],
    vocab: list[str],
    max_len: int = 20
) -> list[str]:
    words = ['<s>', '<s>']
    for _ in range(max_len):
        i = voc2int[words[-2]]
        j = voc2int[words[-1]]
        probs = smoothed[i][j]
        idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
        next_word = vocab[idx]
        if next_word == '</s>':
            break
        words.append(next_word)
    return words[2:]

# ============================================================
# COMPARISON
# ============================================================

if __name__ == "__main__":
    tokens = []
    files = glob("./corpus/*")

    for f in files:
        fcontent = readFile(f)
        tokens.append(tokenize(fcontent))

    vocab = buildVocab(tokens)
    nvocab = len(vocab)

    voc2int = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    print(f"Number of documents: {len(tokens)}")
    print(f"Vocabulary size: {nvocab}")
    print()

    # --------------------------------------------------------
    # Trigram and bigram counts
    # --------------------------------------------------------

    trigram_list, bigram_list = returnTrigramCounts_list(
        tokens,
        voc2int
    )

    trigram_numpy, bigram_numpy = returnTrigramCounts_numpy(
        tokens,
        voc2int
    )

    trigram_list_np = np.array(trigram_list)
    bigram_list_np = np.array(bigram_list)

    # --------------------------------------------------------
    # Compare trigram counts
    # --------------------------------------------------------

    trigram_same = np.array_equal(
        trigram_list_np,
        trigram_numpy
    )

    print("TRIGRAM COUNTS")
    print(f"Same: {trigram_same}")

    if not trigram_same:
        difference = np.abs(
            trigram_list_np - trigram_numpy
        )
        print(f"Maximum difference: {difference.max()}")

    print()

    # --------------------------------------------------------
    # Compare bigram counts
    # --------------------------------------------------------

    bigram_same = np.array_equal(
        bigram_list_np,
        bigram_numpy
    )

    print("BIGRAM COUNTS")
    print(f"Same: {bigram_same}")

    if not bigram_same:
        difference = np.abs(
            bigram_list_np - bigram_numpy
        )
        print(f"Maximum difference: {difference.max()}")

    print()

    # --------------------------------------------------------
    # Smoothed probabilities
    # --------------------------------------------------------

    smoothed_list = laplaceSmoothing_list(
        trigram_list,
        bigram_list,
        voc2int
    )

    smoothed_numpy = laplaceSmoothing_numpy(
        trigram_numpy,
        bigram_numpy,
        voc2int
    )

    smoothed_list_np = np.array(smoothed_list)

    # --------------------------------------------------------
    # Compare smoothed matrices
    # --------------------------------------------------------

    smoothed_same = np.allclose(
        smoothed_list_np,
        smoothed_numpy
    )

    print("SMOOTHED MATRICES")
    print(f"Same: {smoothed_same}")

    if not smoothed_same:
        difference = np.abs(
            smoothed_list_np - smoothed_numpy
        )
        print(f"Maximum difference: {difference.max()}")

    print()

    # --------------------------------------------------------
    # Compare generated outputs
    # --------------------------------------------------------

    random.seed(42)
    ans_list = generate_list(
        smoothed_list,
        voc2int,
        vocab
    )

    random.seed(42)
    ans_numpy = generate_numpy(
        smoothed_numpy,
        voc2int,
        vocab
    )

    print("GENERATED OUTPUT")
    print(f"List implementation:  {' '.join(ans_list)}")
    print(f"NumPy implementation: {' '.join(ans_numpy)}")
    print(f"Same: {ans_list == ans_numpy}")