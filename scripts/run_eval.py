import argparse

from src.evaluation.run_eval import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_set_path", type=str, default="eval/qa_set.jsonl")
    args = parser.parse_args()
    run_eval(args.qa_set_path)


if __name__ == "__main__":
    main()
