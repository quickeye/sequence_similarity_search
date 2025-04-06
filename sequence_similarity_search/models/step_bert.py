import torch
import torch.nn as nn


class StepBERTEncoder(nn.Module):
    def __init__(self, 
                 vocab_sizes: dict,
                 hidden_size: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()

        # Embedding layers per field
        self.recipe_embedding = nn.Embedding(vocab_sizes['recipe'], hidden_size)
        self.eqp_embedding = nn.Embedding(vocab_sizes['eqp_model'], hidden_size)
        self.step_type_embedding = nn.Embedding(vocab_sizes['step_type'], hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        # Optional: step_name embedding if treated as vocab
        self.use_step_name = 'step_name' in vocab_sizes
        if self.use_step_name:
            self.step_name_embedding = nn.Embedding(vocab_sizes['step_name'], hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head for masked prediction (optional for pretraining)
        self.output_head = nn.Linear(hidden_size, vocab_sizes['step_type'])

    def forward(self, inputs, mask_step=False):
        """
        inputs: dict with keys:
            - recipe_ids: (B, L)
            - eqp_ids: (B, L)
            - step_type_ids: (B, L)
            - position_ids: (B, L)
            - step_name_ids: (B, L) [optional]
        """
        x = self.recipe_embedding(inputs['recipe_ids']) \
          + self.eqp_embedding(inputs['eqp_ids']) \
          + self.step_type_embedding(inputs['step_type_ids']) \
          + self.position_embedding(inputs['position_ids'])

        if self.use_step_name and 'step_name_ids' in inputs:
            x += self.step_name_embedding(inputs['step_name_ids'])

        encoded = self.encoder(x)

        if mask_step:
            logits = self.output_head(encoded)
            return logits  # For loss computation during pretraining

        return encoded  # For downstream usage (embeddings)


if __name__ == "__main__":
    # Dummy test
    model = StepBERTEncoder(vocab_sizes={
        'recipe': 100,
        'eqp_model': 100,
        'step_type': 50,
        'step_name': 1000
    })
    inputs = {
        'recipe_ids': torch.randint(0, 100, (2, 64)),
        'eqp_ids': torch.randint(0, 100, (2, 64)),
        'step_type_ids': torch.randint(0, 50, (2, 64)),
        'position_ids': torch.arange(64).unsqueeze(0).repeat(2, 1),
        'step_name_ids': torch.randint(0, 1000, (2, 64))
    }
    out = model(inputs, mask_step=True)
    print("Logits:", out.shape)  # Expected: (B, L, vocab_size of step_type)
