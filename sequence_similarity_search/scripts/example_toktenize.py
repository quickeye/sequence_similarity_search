# script/example_tokenize.py

from sequence_similarity_search.tokenization.vocab import Vocabulary
from sequence_similarity_search.tokenization.tokenizer import SequenceTokenizer

# Simulated input
flows = [
    [{"step_name": "ETCH_A", "recipe": "R1", "eqp_model": "XP500"},
     {"step_name": "CLEAN_B", "recipe": "R1", "eqp_model": "XP510"}],
    [{"step_name": "CVD_X", "recipe": "R2", "eqp_model": "XP600"}]
]

# Build vocabs
step_vocab = Vocabulary()
recipe_vocab = Vocabulary()
eqp_vocab = Vocabulary()

for flow in flows:
    step_vocab.add_tokens_from_iterable([s["step_name"] for s in flow])
    recipe_vocab.add_tokens_from_iterable([s["recipe"] for s in flow])
    eqp_vocab.add_tokens_from_iterable([s["eqp_model"] for s in flow])

# Tokenize
tokenizer = SequenceTokenizer(step_vocab, recipe_vocab, eqp_vocab)
for flow in flows:
    encoded = tokenizer.encode_sequence(flow)
    print(encoded)
