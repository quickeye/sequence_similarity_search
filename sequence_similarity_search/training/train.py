import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from sequence_similarity_search.models.step_bert import StepBERTEncoder
from sequence_similarity_search.training.dataset import MaskedStepDataset
from sequence_similarity_search.tokenization.vocab import Vocabulary
from sequence_similarity_search.tokenization.tokenizer import SequenceTokenizer

def train_model(
    vocab_sizes,
    tokenizer,
    dataset_path,
    batch_size=16,
    num_epochs=10,
    max_len=128,
    lr=1e-4,
    mask_prob=0.15,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    log_dir = f"runs/stepbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Dataset & Dataloader
    dataset = MaskedStepDataset(dataset_path, tokenizer, max_len=max_len, mask_prob=mask_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = StepBERTEncoder(vocab_sizes=vocab_sizes, max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.step_type_vocab["[PAD]"])

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            logits = model(inputs, mask_step=True)  # (B, L, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and "encoder" in name:
                    writer.add_histogram(f\"grads/{name}\", param.grad, epoch)

            optimizer.step()

            total_loss += loss.item()

            writer.add_scalar("Loss/train", total_loss, epoch)

            # Optionally log learning rate
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"LR/group_{i}", param_group['lr'], epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")

    writer.close()

    return model


if __name__ == "__main__":

    # Load vocabularies
    step_type_vocab = Vocabulary.load("data/step_type_vocab.json")
    recipe_vocab = Vocabulary.load("data/recipe_vocab.json")
    eqp_vocab = Vocabulary.load("data/eqp_vocab.json")

    vocab_sizes = {
        "step_type": len(step_type_vocab.token_to_id),
        "recipe": len(recipe_vocab.token_to_id),
        "eqp_model": len(eqp_vocab.token_to_id)
    }

    tokenizer = SequenceTokenizer(
        #step_vocab=None,
        recipe_vocab=recipe_vocab,
        eqp_vocab=eqp_vocab,
        step_type_vocab=step_type_vocab,
        max_len=128
    )

    model = train_model(
        vocab_sizes=vocab_sizes,
        tokenizer=tokenizer,
        dataset_path="data/example_flows.json",
        batch_size=8,
        num_epochs=3000
    )
