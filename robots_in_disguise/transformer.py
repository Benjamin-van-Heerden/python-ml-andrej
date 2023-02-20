import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # how many independent sequences to process in parallel
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # how many training iterations to run
eval_interval = 500 # how often to evaluate the model on the validation set
learning_rate = 1e-3 # how fast should the model learn?
device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available
eval_iters = 200 # how many iterations to run the evaluation for
n_embd = 32 # how many dimensions to use for the embeddings
# ------------------------------

torch.manual_seed(0) # make results reproducible

with open("../tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder -> take string, output list of ints
decode = lambda l: "".join([itos[i] for i in l]) # decoder -> take list of ints, output string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """
    One head of self attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
        # compute the attention scores ("affinity")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # B, T, T
        # mask out the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # apply softmax to get the attention weights
        wei = wei.softmax(dim=-1) # B, T, T
        wei = self.dropout(wei)
        # apply the attention weights to the values
        v = self.value(x) # B, T, C
        return wei @ v # B, T, C

class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    """ simple feed-forward network with ReLU activation """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # also add layer normalization - just like batch norm but over rows not columns
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # also add residual connections to control the flow of gradients
        # communication
        x = x + self.sa_head(self.ln1(x)) # this is a departure from the original paper, this is known as the pre norm formulation
        # computation
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    """
    Model that uses Transformer blocks
    """
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # positional embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both B, T
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # 1, T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # compute the loss
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply the softmax to get probabilities
            probs = logits.softmax(dim=-1) # (B, C)
            # sample from the distribution
            idx_new = torch.multinomial(probs, 1) # (B, 1)
            # append the sampled index to the running sequence
            idx = torch.cat([idx, idx_new], dim=-1)
        return idx

model = TransformerLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
