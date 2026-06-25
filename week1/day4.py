from typing import Iterator


def read_lines(filepath: str) -> Iterator[str]:
    """Yields one raw line at a time. File stays open only while iterating."""
    with open(filepath, "r") as f:
        for line in f:
            yield line


def filter_blank(lines: Iterator[str]) -> Iterator[str]:
    """Drops lines that are empty or whitespace-only."""
    for line in lines:
        if line.strip():
            yield line


def strip_lines(lines: Iterator[str]) -> Iterator[str]:
    """Strips leading/trailing whitespace from each line."""
    for line in lines:
        yield line.strip()


def chunk_lines(lines: Iterator[str], n: int) -> Iterator[list[str]]:
    """Collects lines into fixed-size chunks and yields each batch."""
    chunk: list[str] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:          # yield the final partial chunk, if any
        yield chunk

def build_pipeline(filepath: str, chunk_size: int) -> Iterator[list[str]]:
    """
    Chains all stages into a single lazy pipeline.
    No line is loaded more than once; only one chunk lives in memory at a time.
    """
    lines   = read_lines(filepath)
    cleaned = filter_blank(lines)
    trimmed = strip_lines(cleaned)
    chunks  = chunk_lines(trimmed, chunk_size)
    return chunks

import tempfile, os

sample = "\n  Hello World  \n\nPython Generators\n  \nAre Memory Efficient\nChunk One Done\n\nLine Six\nLine Seven\n"
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
    tmp.write(sample)
    tmp_path = tmp.name

print(f"Processing: {tmp_path}\n")

for i, chunk in enumerate(build_pipeline(tmp_path, chunk_size=2), start=1):
    print(f"Chunk {i}: {chunk}")

os.unlink(tmp_path)