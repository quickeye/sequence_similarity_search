class SequenceTokenizer:
    def __init__(self, 
                 step_vocab,
                 recipe_vocab=None,
                 eqp_vocab=None,
                 step_type_vocab=None,
                 max_len=128):
        self.step_vocab = step_vocab
        self.recipe_vocab = recipe_vocab
        self.eqp_vocab = eqp_vocab
        self.step_type_vocab = step_type_vocab
        self.max_len = max_len

    def encode_sequence(self, steps):
        """
        Tokenizes a flow sequence into vocab IDs, supports multiple vocab fields.
        """
        step_ids = []
        recipe_ids = []
        eqp_ids = []
        type_ids = []
        layer_ids = []

        for step in steps:
            step_ids.append(self.step_vocab[step["step_name"]])

            if self.recipe_vocab:
                recipe_ids.append(self.recipe_vocab[step.get("recipe", "[PAD]")])
            if self.eqp_vocab:
                eqp_ids.append(self.eqp_vocab[step.get("eqp_model", "[PAD]")])
            if self.step_type_vocab:
                type_ids.append(self.step_type_vocab[step.get("step_type", "[PAD]")])

            # Use layer as a float if available (not tokenized)
            layer_ids.append(step.get("layer", 0.0))

        pad_len = max(0, self.max_len - len(step_ids))
        step_ids += [self.step_vocab["[PAD]"]] * pad_len
        recipe_ids += [self.recipe_vocab["[PAD]"]] * pad_len if self.recipe_vocab else []
        eqp_ids += [self.eqp_vocab["[PAD]"]] * pad_len if self.eqp_vocab else []
        type_ids += [self.step_type_vocab["[PAD]"]] * pad_len if self.step_type_vocab else []
        layer_ids += [0.0] * pad_len  # pad layers with 0.0

        return {
            "step_ids": step_ids[:self.max_len],
            "recipe_ids": recipe_ids[:self.max_len] if self.recipe_vocab else None,
            "eqp_ids": eqp_ids[:self.max_len] if self.eqp_vocab else None,
            "step_type_ids": type_ids[:self.max_len] if self.step_type_vocab else None,
            "layer_ids": layer_ids[:self.max_len],
        }
