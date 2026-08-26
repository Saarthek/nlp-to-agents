import numpy as np
from glob import glob
import string
from collections import defaultdict
import math
import random
import time

def readFile(filepath: str) -> str:
    """
    Returns the contents of a file in a string, given its path.
    """
    with open(filepath, "r") as fptr:
        return fptr.read()

def tokenize(doc: str) -> list[str]:
    """
    Returns a list of words given a string with file contents.
    """
    translator = str.maketrans("", "", string.punctuation)
    return doc.strip().lower().translate(translator).split()

# ============================================================
# PURE PYTHON IMPLEMENTATION
# ============================================================

def buildVocabPython(docs: list[list[str]]) -> list[str]:
    """
    Given a list of tokenized documents, construct vocabulary.
    """
    vocab = set()

    for doc in docs:
        vocab.update(doc)

    return sorted(vocab)

def countWordsPython(
    doc: list[str],
    vocab: list[str]
) -> list[int]:
    """
    Given a document and vocabulary, return its count vector.
    Words absent from the vocabulary are ignored.
    """
    counts = defaultdict(int)

    for word in doc:
        if word in vocab:
            counts[word] += 1

    return [
        counts.get(word, 0)
        for word in vocab
    ]

def tfPython(
    countsArr: list[int]
) -> list[float]:
    """
    Given a count vector, return its term-frequency vector.

    TF = 1 + log10(count) for count > 0
    TF = 0 for count = 0
    """
    counts = countsArr.copy()

    for i in range(len(counts)):
        if counts[i] > 0:
            counts[i] = 1 + math.log10(counts[i])

    return counts

def idfPython(
    counts: list[list[int]]
) -> list[float]:
    """
    Given a document-term count matrix, return the IDF vector.

    IDF = log10(N / DF)
    """
    ndocs = len(counts)
    nvocab = len(counts[0])

    idfarr = [0.0] * nvocab

    for i in range(nvocab):
        for j in range(ndocs):
            if counts[j][i] > 0:
                idfarr[i] += 1

    for i in range(nvocab):
        if idfarr[i] > 0:
            idfarr[i] = math.log10(
                ndocs / idfarr[i]
            )

    return idfarr

def tfidfPython(
    tfarr: list[list[float]],
    idfarr: list[float]
) -> list[list[float]]:
    """
    Multiply TF and IDF to produce TF-IDF matrix.
    """
    result = []

    for i in range(len(tfarr)):
        row = []

        for j in range(len(idfarr)):
            row.append(
                tfarr[i][j] * idfarr[j]
            )

        result.append(row)

    return result

def cosineSimPython(
    vec1: list[float],
    vec2: list[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    dot = sum(
        a * b
        for a, b in zip(vec1, vec2)
    )

    norm1 = math.sqrt(
        sum(a * a for a in vec1)
    )

    norm2 = math.sqrt(
        sum(b * b for b in vec2)
    )

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

def queryVectorPython(
    query: str,
    vocab: list[str],
    idfarr: list[float]
) -> list[float]:
    """
    Convert a query into a TF-IDF vector.

    Words absent from the vocabulary are ignored.
    """
    qtokens = tokenize(query)

    qcounts = countWordsPython(
        qtokens,
        vocab
    )

    qtf = tfPython(qcounts)

    return [
        qtf[i] * idfarr[i]
        for i in range(len(vocab))
    ]

def searchPython(
    query: str,
    vocab: list[str],
    idfarr: list[float],
    tfarr: list[list[float]],
    topk: int = 3
) -> list[tuple[int, float]]:
    """
    Search documents using cosine similarity.
    """
    qvec = queryVectorPython(
        query,
        vocab,
        idfarr
    )

    scores = [
        (
            i,
            cosineSimPython(
                qvec,
                tfarr[i]
            )
        )
        for i in range(len(tfarr))
    ]

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:topk]

def runPythonImplementation(
    tokens: list[list[str]]
):
    """
    Run the complete pure Python TF-IDF pipeline.
    """
    vocab = buildVocabPython(tokens)

    freqs = [
        countWordsPython(
            doc,
            vocab
        )
        for doc in tokens
    ]

    tfarr = [
        tfPython(freq)
        for freq in freqs
    ]

    idfarr = idfPython(freqs)

    tfidfarr = tfidfPython(
        tfarr,
        idfarr
    )

    return (
        vocab,
        freqs,
        tfarr,
        idfarr,
        tfidfarr
    )

# ============================================================
# NUMPY IMPLEMENTATION
# ============================================================

def buildVocabNumpy(
    docs: list[list[str]]
) -> list[str]:
    """
    Given a list of tokenized documents, construct vocabulary.
    """
    vocab = set()

    for doc in docs:
        vocab.update(doc)

    return sorted(vocab)

def countWordsNumpy(
    doc: list[str],
    voc2int: dict[str, int]
) -> np.ndarray:
    """
    Given a document and vocabulary mapping,
    return its count vector.

    Words absent from the vocabulary are ignored.
    """
    n = len(voc2int)

    counts = np.zeros(
        n,
        dtype=float
    )

    idxs = [
        voc2int[word]
        for word in doc
        if word in voc2int
    ]

    np.add.at(
        counts,
        idxs,
        1
    )

    return counts

def countMatrixNumpy(
    tokens: list[list[str]],
    voc2int: dict[str, int]
) -> np.ndarray:
    """
    Return a 2D document-term count matrix.
    """
    ndocs = len(tokens)
    nvocab = len(voc2int)

    countMat = np.zeros(
        (ndocs, nvocab),
        dtype=float
    )

    for i, doc in enumerate(tokens):
        countMat[i] = countWordsNumpy(
            doc,
            voc2int
        )

    return countMat

def tfNumpy(
    freqs: np.ndarray
) -> np.ndarray:
    """
    Given a count matrix, calculate TF using NumPy.

    TF = 1 + log10(count) for count > 0
    TF = 0 for count = 0
    """
    result = np.zeros_like(
        freqs,
        dtype=float
    )

    mask = freqs > 0

    result[mask] = (
        1 + np.log10(freqs[mask])
    )

    return result

def idfNumpy(
    counts: np.ndarray,
    ndocs: int
) -> np.ndarray:
    """
    Given a document-term count matrix,
    calculate IDF using NumPy.
    """
    df = np.count_nonzero(
        counts > 0,
        axis=0
    )

    return np.log10(
        ndocs / df
    )

def tfidfNumpy(
    tfarr: np.ndarray,
    idfarr: np.ndarray
) -> np.ndarray:
    """
    Calculate TF-IDF using NumPy broadcasting.
    """
    return tfarr * idfarr

def cosineSimNumpy(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calculate cosine similarity using NumPy.
    """
    dot = vec1.dot(vec2)

    norm1 = np.sqrt(
        vec1.dot(vec1)
    )

    norm2 = np.sqrt(
        vec2.dot(vec2)
    )

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

def queryVectorNumpy(
    query: str,
    voc2int: dict[str, int],
    idfarr: np.ndarray
) -> np.ndarray:
    """
    Convert a query into a TF-IDF vector.

    Words absent from the vocabulary are ignored.
    """
    qtokens = tokenize(query)

    qcounts = countWordsNumpy(
        qtokens,
        voc2int
    )

    qtf = tfNumpy(qcounts)

    return tfidfNumpy(
        qtf,
        idfarr
    )

def searchNumpy(
    query: str,
    voc2int: dict[str, int],
    idfarr: np.ndarray,
    tfarr: np.ndarray,
    topk: int = 3
) -> list[tuple[int, float]]:
    """
    Search documents using cosine similarity.
    """
    qvec = queryVectorNumpy(
        query,
        voc2int,
        idfarr
    )

    scores = [
        (
            i,
            cosineSimNumpy(
                qvec,
                tfarr[i]
            )
        )
        for i in range(len(tfarr))
    ]

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:topk]

def runNumpyImplementation(
    tokens: list[list[str]]
):
    """
    Run the complete NumPy TF-IDF pipeline.
    """
    vocab = buildVocabNumpy(tokens)

    voc2int = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    freqs = countMatrixNumpy(
        tokens,
        voc2int
    )

    tfarr = tfNumpy(freqs)

    idfarr = idfNumpy(
        freqs,
        len(tokens)
    )

    tfidfarr = tfidfNumpy(
        tfarr,
        idfarr
    )

    return (
        vocab,
        freqs,
        tfarr,
        idfarr,
        tfidfarr,
        voc2int
    )

# ============================================================
# OUTPUT HELPERS
# ============================================================

def printMatrix(
    matrix,
    name: str
):
    print(f"\n{name}:")

    for row in matrix:
        print(row)

def compareResults(
    pythonResult,
    numpyResult,
    files: list[str],
    query: str
):
    pythonScores = searchPython(
        query,
        pythonResult[0],
        pythonResult[3],
        pythonResult[4]
    )

    numpyScores = searchNumpy(
        query,
        numpyResult[5],
        numpyResult[3],
        numpyResult[4]
    )

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    print("\nPure Python:")

    for idx, score in pythonScores:
        print(
            f"{files[idx]} -> "
            f"{score:.10f}"
        )

    print("\nNumPy:")

    for idx, score in numpyScores:
        print(
            f"{files[idx]} -> "
            f"{score:.10f}"
        )

    pythonRanking = [
        idx
        for idx, _ in pythonScores
    ]

    numpyRanking = [
        idx
        for idx, _ in numpyScores
    ]

    pythonScoreValues = np.array([
        score
        for _, score in pythonScores
    ])

    numpyScoreValues = np.array([
        score
        for _, score in numpyScores
    ])

    sameRanking = (
        pythonRanking == numpyRanking
    )

    sameScores = np.allclose(
        pythonScoreValues,
        numpyScoreValues,
        rtol=1e-10,
        atol=1e-10
    )

    sameVocabulary = (
        pythonResult[0] == numpyResult[0]
    )

    sameCountMatrix = np.array_equal(
        np.array(pythonResult[1]),
        numpyResult[1]
    )

    sameTF = np.allclose(
        np.array(pythonResult[2]),
        numpyResult[2]
    )

    sameIDF = np.allclose(
        np.array(pythonResult[3]),
        numpyResult[3]
    )

    sameTFIDF = np.allclose(
        np.array(pythonResult[4]),
        numpyResult[4]
    )

    print("\n" + "-" * 70)
    print("CORRECTNESS COMPARISON")
    print("-" * 70)

    print(
        f"Same vocabulary: "
        f"{sameVocabulary}"
    )

    print(
        f"Same count matrix: "
        f"{sameCountMatrix}"
    )

    print(
        f"Same TF matrix: "
        f"{sameTF}"
    )

    print(
        f"Same IDF vector: "
        f"{sameIDF}"
    )

    print(
        f"Same TF-IDF matrix: "
        f"{sameTFIDF}"
    )

    print(
        f"Same ranking: "
        f"{sameRanking}"
    )

    print(
        f"Same scores: "
        f"{sameScores}"
    )

# ============================================================
# BENCHMARK DATA GENERATION
# ============================================================

def generateSyntheticCounts(
    ndocs: int,
    nvocab: int,
    maxCount: int = 5
) -> list[list[int]]:
    """
    Generate a synthetic count matrix using Python lists.
    """
    return [
        [
            random.randint(
                0,
                maxCount
            )
            for _ in range(nvocab)
        ]
        for _ in range(ndocs)
    ]

def generateSyntheticCountsNumpy(
    ndocs: int,
    nvocab: int,
    maxCount: int = 5
) -> np.ndarray:
    """
    Generate a synthetic count matrix using NumPy.
    """
    return np.random.randint(
        0,
        maxCount + 1,
        size=(ndocs, nvocab)
    ).astype(float)

# ============================================================
# BENCHMARK: PURE PYTHON
# ============================================================

def benchmarkPython(
    counts: list[list[int]]
) -> float:
    """
    Benchmark TF + IDF + TF-IDF using pure Python.
    """
    start = time.perf_counter()

    tfarr = [
        tfPython(row)
        for row in counts
    ]

    idfarr = idfPython(
        counts
    )

    tfidfPython(
        tfarr,
        idfarr
    )

    end = time.perf_counter()

    return end - start

# ============================================================
# BENCHMARK: NUMPY
# ============================================================

def benchmarkNumpy(
    counts: np.ndarray
) -> float:
    """
    Benchmark TF + IDF + TF-IDF using NumPy.
    """
    start = time.perf_counter()

    tfarr = tfNumpy(
        counts
    )

    idfarr = idfNumpy(
        counts,
        counts.shape[0]
    )

    tfidfNumpy(
        tfarr,
        idfarr
    )

    end = time.perf_counter()

    return end - start

# ============================================================
# RUN BENCHMARK
# ============================================================

def runBenchmark():
    print("\n\n")
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    print(
        "\nThe benchmark measures only:"
    )

    print(
        "  Count Matrix -> TF -> IDF -> TF-IDF"
    )

    print(
        "\nIt excludes:"
    )

    print(
        "  File reading"
    )

    print(
        "  Tokenization"
    )

    print(
        "  Vocabulary construction"
    )

    print(
        "  Printing"
    )

    print(
        "\nThis gives us a cleaner comparison of "
        "Python loops vs NumPy vectorization."
    )

    benchmarkSizes = [
        (10, 100),
        (100, 1000),
        (500, 2000),
        (1000, 5000),
        (2000, 10000)
    ]

    print("\n")

    print(
        f"{'Documents':>12} "
        f"{'Vocabulary':>12} "
        f"{'Python (s)':>15} "
        f"{'NumPy (s)':>15} "
        f"{'Speedup':>12}"
    )

    print("-" * 70)

    for ndocs, nvocab in benchmarkSizes:
        random.seed(42)
        np.random.seed(42)

        pythonCounts = generateSyntheticCounts(
            ndocs,
            nvocab
        )

        numpyCounts = np.array(
            pythonCounts,
            dtype=float
        )

        pythonTime = benchmarkPython(
            pythonCounts
        )

        numpyTime = benchmarkNumpy(
            numpyCounts
        )

        speedup = (
            pythonTime / numpyTime
            if numpyTime > 0
            else float("inf")
        )

        print(
            f"{ndocs:>12} "
            f"{nvocab:>12} "
            f"{pythonTime:>15.6f} "
            f"{numpyTime:>15.6f} "
            f"{speedup:>11.2f}x"
        )

    print("\n" + "-" * 70)

    print(
        "Speedup = Pure Python time / NumPy time"
    )

    print(
        "Higher speedup means NumPy provides "
        "a larger performance advantage."
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    files = glob("./corpus/*")

    if not files:
        print("No files found in ./corpus/")
        exit()

    tokens = []

    for filepath in files:
        content = readFile(filepath)

        tokens.append(
            tokenize(content)
        )

    print(
        f"Found {len(files)} documents."
    )

    query = input(
        "\nPlease enter your query: "
    )

    # ========================================================
    # PURE PYTHON
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNNING PURE PYTHON IMPLEMENTATION")
    print("=" * 70)

    pythonResult = runPythonImplementation(
        tokens
    )

    pythonVocab = pythonResult[0]
    pythonFreqs = pythonResult[1]
    pythonTF = pythonResult[2]
    pythonIDF = pythonResult[3]
    pythonTFIDF = pythonResult[4]

    print(
        f"\nNumber of documents: "
        f"{len(tokens)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(pythonVocab)}"
    )

    print(
        f"\nVocabulary:\n"
        f"{pythonVocab}"
    )

    printMatrix(
        pythonFreqs,
        "Pure Python Count Matrix"
    )

    printMatrix(
        pythonTF,
        "Pure Python TF Matrix"
    )

    print(
        "\nPure Python IDF Vector:"
    )

    print(pythonIDF)

    printMatrix(
        pythonTFIDF,
        "Pure Python TF-IDF Matrix"
    )

    # ========================================================
    # NUMPY
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNNING NUMPY IMPLEMENTATION")
    print("=" * 70)

    numpyResult = runNumpyImplementation(
        tokens
    )

    numpyVocab = numpyResult[0]
    numpyFreqs = numpyResult[1]
    numpyTF = numpyResult[2]
    numpyIDF = numpyResult[3]
    numpyTFIDF = numpyResult[4]

    print(
        f"\nNumber of documents: "
        f"{len(tokens)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(numpyVocab)}"
    )

    print(
        f"\nVocabulary:\n"
        f"{numpyVocab}"
    )

    printMatrix(
        numpyFreqs,
        "NumPy Count Matrix"
    )

    printMatrix(
        numpyTF,
        "NumPy TF Matrix"
    )

    print(
        "\nNumPy IDF Vector:"
    )

    print(numpyIDF)

    printMatrix(
        numpyTFIDF,
        "NumPy TF-IDF Matrix"
    )

    # ========================================================
    # CORRECTNESS COMPARISON
    # ========================================================

    compareResults(
        pythonResult,
        numpyResult,
        files,
        query
    )

    # ========================================================
    # PERFORMANCE BENCHMARK
    # ========================================================

    runBenchmark()