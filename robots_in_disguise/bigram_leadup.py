#%%
with open("./tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])
# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print("".join(chars))
# %%
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x] # encode a string to a list of integers
decode = lambda x: "".join([itos[ch] for ch in x]) # decode a list of integers to a string

print(encode("hello"))
print(decode(encode("hello")))
# %%
import torch
data = torch.tensor(encode(text), dtype=torch.int64)
print(data.shape, data.dtype)
print(data[:100])
# %%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# %%

# block size is the maximum length of a sequence
block_size = 8
train_data[:block_size + 1]

# %%

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t:t]
    print(f"When input is {context}, the target is {target}")
# %%
torch.manual_seed(0)
batch_size = 4

def get_batch(split):
    # genereate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print(xb.shape, yb.shape)
print(xb)
print(yb)

# %%
for b in range(batch_size): # batch dimension
    for t in range(block_size): # sequence (time) dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context}, the target is {target}")
# %%
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, source, targets=None):
        # source and targets are both (B, T) tensors

        logits = self.token_embedding_table(source) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # pytorch expects F.cross_entropy to be called with (N, C)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, source, max_new_tokens=100):
        # source is a (B, T) tensor and max_new_tokens is an integer
        # returns a (B, T + max_new_tokens) tensor
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(source)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get a probability distribution
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            new_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the new tokens to the source
            source = torch.cat([source, new_token], dim=1) # (B, T_current + 1)
        return source


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# %%
seed_for_gen = torch.zeros((1, 1), dtype=torch.int64)
print(decode(m.generate(seed_for_gen)[0].tolist()))
# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# %%
batch_size = 32
for step in range(10000):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, loss {loss.item()}")
# %%
seed_for_gen = torch.zeros((1, 1), dtype=torch.int64)
print(decode(m.generate(seed_for_gen, max_new_tokens=500)[0].tolist()))
# %%

# now the tokens need to start talking to each other to figure out what we have in the context