import argparse

from src.index.build import build_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")
    args = parser.parse_args()
    build_indices(args.chunks_path)


if __name__ == "__main__":
    main()
