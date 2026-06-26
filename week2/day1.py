from abc import ABC, abstractmethod


# ── Interface ────────────────────────────────────────────────────────────────

class Chunker(ABC):
    """Common interface every chunking strategy must satisfy."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of non-empty string chunks."""

    # Lets client code print the strategy name without reaching into __class__
    def __str__(self) -> str:
        return self.__class__.__name__


# ── Concrete strategies ───────────────────────────────────────────────────────

class FixedSizeChunker(Chunker):
    """Slice the text into fixed-width character windows."""

    def __init__(self, size: int = 20) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size = size

    def chunk(self, text: str) -> list[str]:
        return [text[i : i + self.size] for i in range(0, len(text), self.size)]


class SentenceChunker(Chunker):
    """Split on full-stops; empty fragments are discarded."""

    def chunk(self, text: str) -> list[str]:
        return [s.strip() for s in text.split(".") if s.strip()]


class RecursiveChunker(Chunker):
    """Group words into fixed-size word windows (default: 5 words)."""

    def __init__(self, words_per_chunk: int = 5) -> None:
        if words_per_chunk < 1:
            raise ValueError(f"words_per_chunk must be >= 1, got {words_per_chunk}")
        self.words_per_chunk = words_per_chunk

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + self.words_per_chunk])
            for i in range(0, len(words), self.words_per_chunk)
        ]


# ── Context ───────────────────────────────────────────────────────────────────

class TextProcessor:
    """Runs whichever Chunker strategy it is given — nothing more."""

    def __init__(self, chunker: Chunker) -> None:
        self._chunker = chunker

    def process(self, text: str) -> list[str]:
        return self._chunker.chunk(text)

    # Delegates name resolution to the strategy — client stays strategy-agnostic
    def strategy_name(self) -> str:
        return str(self._chunker)


# ── Client code (identical regardless of strategy) ────────────────────────────

if __name__ == "__main__":
    text = (
        "Python is great for AI. "
        "Design patterns improve maintainability. "
        "Strategy pattern allows flexibility."
    )

    strategies: list[Chunker] = [
        FixedSizeChunker(25),
        SentenceChunker(),
        RecursiveChunker(),
    ]

    for strategy in strategies:
        processor = TextProcessor(strategy)          # ← identical for every strategy
        print(f"\nUsing {processor.strategy_name()}")
        for chunk in processor.process(text):        # ← identical for every strategy
            print(chunk)