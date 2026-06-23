import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads                                                                                                  
        self.qkv = nn.Linear(d_model, d_model * 3)                                                                                     
        self.proj = nn.Linear(d_model, d_model)                                                                                        
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, x):                                                                                                              
        B, L, D = x.shape                                                                                                              
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)

        # Causal mask: upper triangle contains future tokens; fill with -inf                                                           
        mask = torch.triu(torch.ones((L, L)), diagonal=1).bool().to(x.device)

        q, k, v = qkv[0], qkv[1], qkv[2]                                                                                               
        attn = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k)).to(x.device)                                            
        attn = attn.masked_fill(mask, float("-inf"))                                                                                   
        attn = F.softmax(attn, dim=-1)

        out = (v @ attn).transpose(-2, -1).reshape(B, L, D)   # concatenate heads back together                                                                                    
        return self.proj(self.dropout(out))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_heads = n_heads

        # SwiGLU MLP: three projections (two for the gating form, one to project back)                                                  
        hidden_dim = int(d_model * 3 / 2)  # force integer size                                                                        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # gate                                                                      
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)   # linear                                                                       
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)    # final projection

    def forward(self, x):                                                                                                              
        x = x + self.attn(self.ln1(x))                                                                                                 
        gate = F.silu(self.w1(x))   # [B, L, 3d/2]                                                                                       
        linear = self.w2(x)         # [B, L, 3d/2]

        out = self.w3((gate * linear).reshape(-1, self.hidden_dim))   # element-wise mult then project back                                                   
        return x + out.view(*x.shape[:-1], self.d_model)

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, layers=12):
        super().__init__()

        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")  # byte-level BPE tokenizer                                                                  
        self.wpe = nn.Parameter(torch.randn(2048, d_model))     # learned position embeddings                                                                
        self.tok_emb = nn.Embedding(vocab_size, d_model)                                                                               
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(layers)])                                                 
        self.lnF = nn.LayerNorm(d_model)                                                                                               
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

def forward(self, idx):                                                                                                            
    if not isinstance(idx, torch.Tensor):      # handle raw list input as a fallback too                                                              
        idx = torch.tensor([idx], dtype=torch.long)

    x = self.tok_emb(idx).view(-1, -1)          # collapse any rank → [B, T]                                                                  
    for block in self.blocks: 
        x = block(x)
    return self.lm_head(self.lnF(x))

def generate(self, prompt, max=50):                                                                                                 
    with torch.no_grad():
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long)                                                           
        for _ in range(max):
            logits = self.forward(idx[:, -1:])      # predict next token from last position
            p = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(p[0], 1).unsqueeze(0)   # random sampling (temperature=1)
            idx = torch.cat([idx, next_id], dim=1)

        return self.tok.decode(idx[0].tolist())

#Load WikiText-103 and tokenize it using the same byte-level BPE as before

ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def train(model, epochs=4, lr=6e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(ds, desc=f"Epoch {epoch+1}"):
            tokens = tokenizer.encode(batch["text"])
            if len(tokens) < 2: continue

        x = torch.tensor([tokens], dtype=torch.long).to("cuda") 
        y = x[0][1:].unsqueeze(0)      # target is the next token (shift right by one)
        inputs = x[:, :-1].unsqueeze(1) # input is all but last

        optimizer.zero_grad()                                                                                                     
        logits = model(inputs)           # shape [B, T+1, 50257]
        loss = torch.nn.functional.cross_entropy(logits[..., :-1], y).mean()   # exclude the extra predicted token at the end

        loss.backward()                                                                                                          
        optimizer.step()                                                                                                         
        total_loss += loss.item()

    print(f"Epoch {epoch+1} avg loss: {total_loss/len(ds):.4f}")

model = GPT2().to("cuda")
train(model)
