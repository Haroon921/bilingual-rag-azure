"""Ask from the terminal.   python -m src.cli "How many days of annual leave do I get?" """
from __future__ import annotations

import sys

from .rag import RagPipeline


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m src.cli "your question"')
    a = RagPipeline().ask(" ".join(sys.argv[1:]))
    print(a.text)
    if a.cited:
        print("\nSources:")
        for s in a.cited:
            print(f"  [{s.n}] {s.title} ({s.source})")
    print(f"\n[{a.language} | {a.latency_ms:.0f} ms | {a.prompt_tokens}+{a.completion_tokens} tokens]")


if __name__ == "__main__":
    main()
