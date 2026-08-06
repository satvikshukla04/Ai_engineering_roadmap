import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK sentence tokenizer data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# NOTE: This sandbox has no internet access to huggingface.co, so
# langchain_experimental.SemanticChunker + HuggingFaceEmbeddings can't
# download `all-MiniLM-L6-v2`. Strategy 4 below reimplements the same
# percentile-breakpoint semantic chunking algorithm locally, using
# TF-IDF vectors + cosine similarity instead of a neural embedding
# model. Swap `embed_sentences()` for a HuggingFaceEmbeddings call if
# you're running this somewhere with internet access, to get true
# semantic (rather than lexical) similarity.

# 1. Prepare a Sample Document
# We'll use a dense, multi-topic text to see how the splitters react.
SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information. 
When users ask a question, the system searches a database for relevant context, appends it to the prompt, and sends it to the LLM. 
This prevents hallucinations and provides sources for the generated answers.

Chunking is a critical step in this process. You cannot embed a 50-page document as one vector — the embedding averages all meaning, losing specificity. 
You cannot embed every sentence individually — too many vectors, retrieval noise. Chunking finds the sweet spot: pieces large enough to contain a complete thought, small enough to be semantically specific. 

Python is a versatile programming language used widely in data science. It features a simple syntax and a massive ecosystem of libraries. 
Libraries like Pandas, NumPy, and Scikit-Learn make it the go-to choice for machine learning engineers. 
Unlike compiled languages like C++, Python is interpreted, which makes debugging easier but execution slightly slower.

Space exploration has seen massive advancements in the 21st century. Reusable rockets have drastically lowered the cost of reaching orbit. 
Organizations like NASA and SpaceX are currently planning crewed missions to Mars. 
The James Webb Space Telescope continues to provide unprecedented images of deep space, fundamentally changing our understanding of the early universe.
""" * 5  # Multiply to create a reasonably sized document for statistics


def analyze_chunks(chunks, strategy_name):
    """Helper function to calculate stats for a list of chunks."""
    lengths = [len(c) for c in chunks]
    return {
        "Strategy": strategy_name,
        "Total Chunks": len(chunks),
        "Avg Size (chars)": round(np.mean(lengths), 2),
        "Min Size": min(lengths),
        "Max Size": max(lengths),
        "Lengths": lengths,
    }


def embed_sentences(sentences):
    """TF-IDF sentence embeddings (stand-in for a neural embedding model)."""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences).toarray()


def semantic_chunk(text, breakpoint_percentile=80):
    """
    Group sentences into chunks, splitting at points where semantic
    similarity between consecutive sentences drops into the bottom
    `100 - breakpoint_percentile` percentile of all consecutive-pair
    similarities. Mirrors SemanticChunker's percentile-threshold approach.
    """
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    if len(sentences) < 2:
        return sentences

    embeddings = embed_sentences(sentences)
    sims = [
        cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(sentences) - 1)
    ]

    threshold = np.percentile(sims, 100 - breakpoint_percentile)
    split_points = {i + 1 for i, s in enumerate(sims) if s <= threshold}

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if i in split_points:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    results = []

    # Strategy 1: Fixed-size chunking (Blind Character Splitting)
    # Note: We use an empty separator to force splitting exactly at the chunk size.
    fixed_splitter = CharacterTextSplitter(separator="", chunk_size=300, chunk_overlap=50)
    fixed_chunks = fixed_splitter.split_text(SAMPLE_TEXT)
    results.append(analyze_chunks(fixed_chunks, "Fixed-Size (Char)"))

    # Strategy 2: Recursive Character Text Splitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    recursive_chunks = recursive_splitter.split_text(SAMPLE_TEXT)
    results.append(analyze_chunks(recursive_chunks, "Recursive Character"))

    # Strategy 3: Sentence-Aware Splitting (NLTK)
    sentences = sent_tokenize(SAMPLE_TEXT)
    sentence_chunks = []
    # Group every 3 sentences with an overlap of 1 sentence
    chunk_size = 3
    overlap = 1
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk not in sentence_chunks:  # Basic deduplication for tail end
            sentence_chunks.append(chunk)
    results.append(analyze_chunks(sentence_chunks, "Sentence-Aware (NLTK)"))

    # Strategy 4: Semantic Chunking
    # Embeds each sentence, then breaks the text wherever the similarity
    # between consecutive sentences drops sharply (same percentile-breakpoint
    # logic as langchain_experimental's SemanticChunker).
    print("Computing sentence embeddings for Semantic Chunking...")
    semantic_chunks = semantic_chunk(SAMPLE_TEXT, breakpoint_percentile=80)
    results.append(analyze_chunks(semantic_chunks, "Semantic Chunking"))

    # Print Report to Console
    print("\n--- CHUNKING ANALYSIS REPORT ---")
    print(f"{'Strategy':<22} | {'Chunks':<8} | {'Avg Size':<10} | {'Min':<6} | {'Max':<6}")
    print("-" * 60)
    for res in results:
        print(f"{res['Strategy']:<22} | {res['Total Chunks']:<8} | {res['Avg Size (chars)']:<10} | {res['Min Size']:<6} | {res['Max Size']:<6}")

    # Plot Distribution
    plt.figure(figsize=(12, 6))
    for res in results:
        plt.hist(res['Lengths'], alpha=0.5, label=res['Strategy'], bins=15)

    plt.title("Chunk Size Distribution by Strategy")
    plt.xlabel("Chunk Size (Characters)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)

    # Save the plot
    plt.savefig("chunk_distribution.png")
    print("\nPlot saved as 'chunk_distribution.png'")

    return results


if __name__ == "__main__":
    main()