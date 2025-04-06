import json
import torch
from torch.utils.data import Dataset
import random


class MaskedStepDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128, mask_prob=0.15):
        with open(json_path, "r") as f:
            all_flows = json.load(f)

        # Flatten flows from each prefix into a single list
        self.sequences = []
        for flow in all_flows.values():
            self.sequences.append(flow)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.step_type_vocab["[MASK]"]
        self.pad_token_id = tokenizer.step_type_vocab["[PAD]"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        flow = self.sequences[idx]
        tokens = self.tokenizer.encode_sequence(flow)

        step_type_ids = tokens["step_type_ids"]
        labels = step_type_ids.copy()

        # Apply masking for MLM
        for i in range(len(step_type_ids)):
            if step_type_ids[i] == self.pad_token_id:
                continue
            if random.random() < self.mask_prob:
                step_type_ids[i] = self.mask_token_id  # Replace with [MASK]
            else:
                labels[i] = self.pad_token_id  # Don't compute loss for unmasked

        return {
            "recipe_ids": torch.tensor(tokens["recipe_ids"], dtype=torch.long),
            "eqp_ids": torch.tensor(tokens["eqp_ids"], dtype=torch.long),
            "step_type_ids": torch.tensor(step_type_ids, dtype=torch.long),
            "position_ids": torch.tensor(list(range(self.max_len)), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
