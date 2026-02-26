#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

print("""Building Transformer Models from Scratch with PyTorch

https://machinelearningmastery.com/building-transformer-models-from-scratch-with-pytorch-10-day-mini-course/
""")


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 01: Getting the Data")

import os
import requests

DATASOURCE = {
    "memoirs_of_grant": "https://www.gutenberg.org/ebooks/4367.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
    "origin_of_species": "https://www.gutenberg.org/ebooks/2009.txt.utf-8",
    "makers_of_many_things": "https://www.gutenberg.org/ebooks/28569.txt.utf-8",
    "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
    "economic_peace": "https://www.gutenberg.org/ebooks/15776.txt.utf-8",
    "the_great_war_3": "https://www.gutenberg.org/ebooks/29265.txt.utf-8",
    "elements_of_style": "https://www.gutenberg.org/ebooks/37134.txt.utf-8",
    "problem_of_philosophy": "https://www.gutenberg.org/ebooks/5827.txt.utf-8",
    "nights_in_london": "https://www.gutenberg.org/ebooks/23605.txt.utf-8",
}
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)
        print(f"[Download] {filename}.txt")


# Read and preprocess the text
def preprocess_gutenberg(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract the main content
    text = text[start:end].strip()

    # Basic preprocessing
    # Remove multiple newlines and spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text


def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        text = preprocess_gutenberg(f"{filename}.txt")
        all_text.append(text)
    return all_text


text = get_dataset_text()


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 02: Train a Tokenizer for Your Language Model")

import tokenizers

tokenizer_file = "gutenberg_tokenizer.json"

if not os.path.exists(tokenizer_file):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[pad]", "[eos]"],
        show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    # Save the trained tokenizer
    tokenizer.save(tokenizer_file, pretty=True)
    print(f"[Save] {tokenizer_file}")

print(f"[Load] {tokenizer_file}")
tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 03: Positional Encoding")

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())  # [max_seq_len, dim]
        self.register_buffer("sin", sinusoid_inp.sin())  # [max_seq_len, dim]

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)  # [1, seq_len, 1, dim]
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)  # [1, seq_len, 1, dim]
        return apply_rotary_pos_emb(x, cos, sin)


sequence = torch.randn(1, 10, 4, 128)
rope = RotaryPositionalEncoding(128)
new_sequence = rope(sequence)


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 04: Grouped Query Attention")

# batch_size, seq_len, hidden_dim = x.shape

# q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
# k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
# v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
# out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

# q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
# k = k_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
# v = v_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
# output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
# output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim).contiguous()
# output = out_proj(q)


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 05: Causal Mask")

# N = 8
# mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
# print(mask)


def create_causal_mask(seq_len, device, dtype=torch.float32):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
    return mask


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 06: Mixture of Expert Models")


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Create expert networks
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, then compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)

        # Select top-k experts, then softmax output probabilities will sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Allocate output tensor
        output = torch.zeros(batch_size * seq_len, hidden_dim,
                             device=hidden_states.device,
                             dtype=hidden_states.dtype)

        # Process through selected experts
        unique_experts = torch.unique(top_k_indices)
        for i in unique_experts:
            expert_id = int(i)
            # token_mask (boolean tensor) = which token of the input should use this expert
            # token_mask shape: (batch_size * seq_len,)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=1)
            assert token_mask.any(), f"Expecting some tokens using expert {expert_id}"

            # select tokens, apply the expert, then add to the output
            expert_input = hidden_states_reshaped[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1)       # shape: (N, 1)
            expert_output = self.experts[expert_id](expert_input) # shape: (N, hidden_dim)
            output[token_mask] += expert_output * expert_weight

        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_dim)
        return output


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 07: RMS Norm and Skip Connections")

# rms_norm = nn.RMSNorm(hidden_dim)
# output_rms = rms_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, moe_experts, moe_topk, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = MoELayer(hidden_dim, 4 * hidden_dim, moe_experts, moe_topk)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 08: The Complete Transformer Model")

model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "moe_experts": 8,
    "moe_topk": 3,
    "max_seq_len": 512,
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}


class TextGenerationModel(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 moe_experts, moe_topk, max_seq_len, vocab_size, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, moe_experts, moe_topk, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)


model = TextGenerationModel(**model_config)


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 09: Training the Model")

import torch.optim as optim
import tqdm


class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len=512):
        self.seq_len = seq_len
        # Encode the entire text
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

BATCH_SIZE = 4  # if encounter OOM error
# BATCH_SIZE = 32  # for better training, but may require a GPU with more memory
text = "\n".join(get_dataset_text())
dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).to(torch.bfloat16)


N_EPOCHS = 2
LR = 0.0005
WARMUP_STEPS = 2000
CLIP_NORM = 6.0

optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))

# Learning rate scheduling
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_STEPS])

print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
best_loss = float('inf')

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    for x, y in progress_bar:
        x = x.to(device)
        y = y.to(device)

        # Create causal mask
        mask = create_causal_mask(x.shape[1], device, torch.bfloat16)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x, mask.unsqueeze(0))

        # Compute loss
        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CLIP_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        # Show loss in tqdm
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")

    # Save checkpoint if loss improved
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "textgen_model.pth")


# --------------------------------------------------
print("\n" + "-" * 50)
print("Lesson 10: Using the Model")


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt, set tensor to batch size of 1
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions for the next token as the last element of the output
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Stop if we predict the end token
            if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())


# Test the model with some prompts
test_prompts = [
    "Once upon a time,",
    "We the people of the",
    "In the beginning was the",
]

print("\nGenerating sample texts:")
for prompt in test_prompts:
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 80)
