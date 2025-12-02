# mini_tokenizer.py
#This is Byte pair encoding (BPE)
from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


@dataclass
class BPETokenizerConfig:
    vocab_size: int = 4096 #  this is where we defined the vocab size
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = SPECIAL_TOKENS.copy()


class BPETokenizer:
    """
    Very small BPE tokenizer implementation.

    Notes:
    - Trains on whitespace-separated words.
    - Uses '</w>' end-of-word marker internally (not exposed to the user).
    - Special tokens are always kept at fixed ids at the start of the vocab.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        merges: List[Tuple[str, str]],
        special_tokens: Dict[str, str],
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.merges = merges
        self.special_tokens = special_tokens

        # Precompute merge ranks for fast lookup during encoding
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Convenience shortcuts
        self.pad_id = self.token_to_id[self.special_tokens["pad"]]
        self.unk_id = self.token_to_id[self.special_tokens["unk"]]
        self.bos_id = self.token_to_id[self.special_tokens["bos"]]
        self.eos_id = self.token_to_id[self.special_tokens["eos"]]

    # ---------- Public API ----------

    @classmethod
    def train_from_text(
        cls,
        text: str,
        config: BPETokenizerConfig,
    ) -> "BPETokenizer":
        """
        Learn a BPE vocabulary from raw text.
        """
        # 1) Build initial word -> count vocab with chars + '</w>'
        words = text.strip().split()
        vocab = Counter([" ".join(list(w) + ["</w>"]) for w in words])

        # Start with character vocabulary
        chars = set()
        for word in vocab:
            chars.update(word.split())

        special_tokens = config.special_tokens
        merges: List[Tuple[str, str]] = []

        # Reserve ids for special tokens first
        token_to_id: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        next_id = 0
        for name in ["pad", "unk", "bos", "eos"]:
            tok = special_tokens[name]
            token_to_id[tok] = next_id
            id_to_token[next_id] = tok
            next_id += 1

        # Add initial characters to vocab
        for ch in sorted(chars):
            if ch in token_to_id:
                continue
            token_to_id[ch] = next_id
            id_to_token[next_id] = ch
            next_id += 1

        # 2) Learn merges until we hit vocab_size
        # Each merge adds a new token
        def get_stats(vocab_counter: Counter) -> Counter:
            pairs = Counter()
            for word, freq in vocab_counter.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pairs[pair] += freq
            return pairs

        def merge_vocab(
            pair: Tuple[str, str],
            vocab_counter: Counter,
        ) -> Counter:
            bigram = " ".join(pair)
            replacement = "".join(pair)
            new_vocab = Counter()
            for word, freq in vocab_counter.items():
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] += freq
            return new_vocab

        # Number of new tokens we’re allowed to add
        max_merges = config.vocab_size - len(token_to_id)

        for _ in range(max_merges):
            stats = get_stats(vocab)
            if not stats:
                break

            best_pair, best_freq = stats.most_common(1)[0]
            if best_freq < 2:
                # No point merging pairs that barely occur
                break

            # Register the merged token
            merged_token = "".join(best_pair)
            if merged_token not in token_to_id:
                token_to_id[merged_token] = next_id
                id_to_token[next_id] = merged_token
                next_id += 1

            merges.append(best_pair)
            vocab = merge_vocab(best_pair, vocab)

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=special_tokens,
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text into token ids using learned BPE merges.

        If BPE produces any symbol that isn't in the vocabulary
        (which can happen with edge cases), we fall back to a
        simple character-level encoding for that word.
        """
        words = text.strip().split()
        ids: List[int] = []

        if add_special_tokens:
            ids.append(self.bos_id)

        for w in words:
            # First try normal BPE
            word_tokens = self._bpe_word(w)

            # If any token is unknown, fall back to pure char-level for this word
            if any(t not in self.token_to_id for t in word_tokens):
                symbols = list(w) + ["</w>"]
                word_tokens = symbols

            for t in word_tokens:
                tid = self.token_to_id.get(t, self.unk_id)
                ids.append(tid)

        if add_special_tokens:
            ids.append(self.eos_id)

        return ids


    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token ids back to text.

        Strategy:
        - Map ids back to token strings.
        - Drop special tokens if requested.
        - Concatenate everything, then treat '</w>' as word boundaries.
        """
        pieces: List[str] = []

        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.special_tokens["unk"])

            if skip_special_tokens and tok in {
                self.special_tokens["pad"],
                self.special_tokens["bos"],
                self.special_tokens["eos"],
            }:
                # I deliberately do NOT drop <unk> here so unknown pieces
                # stay visible in the decoded text.
                continue

            pieces.append(tok)

        # Join all BPE pieces, then turn end-of-word markers into spaces
        text = "".join(pieces)
        text = text.replace("</w>", " ")

        # Normalise whitespace a bit
        text = " ".join(text.split())
        return text


    def save(self, path: str) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        token_to_id = {k: int(v) if isinstance(v, str) and v.isdigit() else v
                       for k, v in data["token_to_id"].items()}
        # id_to_token keys were stringified
        id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        merges = [tuple(p) for p in data["merges"]]
        special_tokens = data["special_tokens"]

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=special_tokens,
        )

    # ---------- Internal helpers ----------

    def _bpe_word(self, word: str) -> List[str]:
        """
        Apply BPE merges to a single word.
        """
        # Start with characters + end-of-word marker
        symbols = list(word) + ["</w>"]
        if not self.merges:
            return symbols

        # We apply merges greedily based on merge_ranks
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate = None
            best_rank = None

            for p in pairs:
                rank = self.merge_ranks.get(p)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    candidate = p

            if candidate is None:
                break

            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == candidate[0]
                    and symbols[i + 1] == candidate[1]
                ):
                    new_symbols.append("".join(candidate))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols
