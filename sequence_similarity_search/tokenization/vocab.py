# sequence_similarity_search/tokenization/vocab.py

from collections import defaultdict
import json

class Vocabulary:
    def __init__(self, special_tokens=None):
        self.token_to_id = {}
        self.id_to_token = {}
        self.frozen = False

        self.special_tokens = special_tokens or ["[PAD]", "[MASK]", "[UNK]"]
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def add_tokens_from_iterable(self, items):
        for item in items:
            self.add_token(item)

    def __getitem__(self, token):
        return self.token_to_id.get(token, self.token_to_id["[UNK]"])

    def get_token(self, idx):
        return self.id_to_token.get(idx, "[UNK]")

    def to_dict(self):
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            token_to_id = json.load(f)
        vocab = cls()
        vocab.token_to_id = token_to_id
        vocab.id_to_token = {idx: token for token, idx in token_to_id.items()}
        return vocab
