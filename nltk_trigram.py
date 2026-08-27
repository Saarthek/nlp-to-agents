from glob import glob
import string
import random
import nltk
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

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
    return ans

if __name__ == "__main__":
    tokens = []
    files = glob("./corpus/*")

    for f in files:
        fcontent = readFile(f)
        tokens.append(tokenize(fcontent))

    n = 3

    train_data, vocabulary = padded_everygram_pipeline(
        n,
        tokens
    )

    model = Laplace(n)
    model.fit(train_data, vocabulary)

    print("Vocabulary:")
    print(model.vocab)
    print(f"Vocabulary size: {len(model.vocab)}")
    print()

    print("Generation:")

    random.seed(42)

    generated = model.generate(
        num_words=20,
        text_seed=["<s>", "<s>"]
    )

    ans = []

    for word in generated:
        if word == "</s>":
            break
        ans.append(word)

    print(" ".join(ans))