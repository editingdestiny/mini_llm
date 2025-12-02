# train_tokenizer.py
import argparse
from pathlib import Path

from mini_tokenizer import BPETokenizer, BPETokenizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/pretrain.txt",
        help="Path to raw training text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer.json",
        help="Where to write the trained tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4096,
        help="Target vocabulary size including special tokens.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input text not found: {input_path}")

    print(f"Reading text from {input_path} ...")
    text = input_path.read_text(encoding="utf-8")

    cfg = BPETokenizerConfig(vocab_size=args.vocab_size)
    print(f"Training BPE tokenizer with vocab size {cfg.vocab_size} ...")
    tokenizer = BPETokenizer.train_from_text(text, cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Saved tokenizer to {output_path}")


if __name__ == "__main__":
    main()
