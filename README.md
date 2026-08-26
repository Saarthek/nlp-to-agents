nlp-to-agents
A hands-on repository tracking my journey from foundational Natural Language Processing (NLP) concepts to building autonomous LLM agents from scratch.

About
This repository serves as a personal learning lab to deeply understand the mechanics behind modern NLP, Large Language Models (LLMs), and AI agents by building key components step-by-step—moving from raw implementations to library-based models.

Repository Structure
corpus/: Sample text data and toy corpora used for testing implementations.

tfidf.py: Pure Python implementation of TF-IDF without external numerical libraries.

np_tfidf.py: Vectorized TF-IDF implementation using NumPy for optimized matrix operations.

sklearntfidf.py: Standard implementation using scikit-learn's TfidfVectorizer for comparison.

merge_tfidf.py: Integration and test script to evaluate and compare TF-IDF outputs across implementations.

🛠️ Getting Started
Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

pip install numpy scikit-learn

Usage
Run basic TF-IDF (Pure Python):

python tfidf.py

Run NumPy vectorized TF-IDF:

python np_tfidf.py

Run scikit-learn benchmark:

python sklearntfidf.py

Verify outputs across approaches:

python merge_tfidf.py

Roadmap
[x] Basic NLP Fundamentals (TF-IDF, Tokenization)

[ ] Word Embeddings & Vector Stores (Word2Vec, Cosine Similarity)

[ ] Language Modeling Foundations & Transformer Architectures

[ ] Prompt Engineering & LLM API Integration

[ ] Autonomous Agent Workflows & Tool Use
