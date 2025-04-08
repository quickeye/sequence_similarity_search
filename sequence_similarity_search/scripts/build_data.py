import os
import json
from collections import defaultdict
from sequence_similarity_search.scripts.generate_sample_flows import generate_multiple_flows

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]"]

def extract_vocab(flows, field):
    vocab = set()
    for flow in flows.values():
        for step in flow:
            if field in step:
                vocab.add(step[field])
    vocab = SPECIAL_TOKENS + sorted(list(vocab))
    return {token: i for i, token in enumerate(vocab)}

def build_vocab_files(example_flows_path, output_dir):
    with open(example_flows_path, "r") as f:
        flows = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for field, filename in [
        ("step_type", "step_type_vocab.json"),
        ("recipe", "recipe_vocab.json"),
        ("eqp_model", "eqp_vocab.json")
    ]:
        vocab = extract_vocab(flows, field)
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"✅ Saved {filename} with {len(vocab)} entries")

def build_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print("🚀 Generating sample flows...")
    generate_multiple_flows(out_file=os.path.join(data_dir, "example_flows.json"))

    print("📦 Building vocabulary files...")
    build_vocab_files(
        example_flows_path=os.path.join(data_dir, "example_flows.json"),
        output_dir=data_dir
    )
    print("✅ Done. Ready for training!")

if __name__ == "__main__":
    build_data()
