import torch
import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F

embed_size = 384
block_size = 256
dropout = 0.2
n_layer = 6
n_head = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

vocab = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '£', 'º', '½', 'É', 'à', 'â', 'æ', 'ç', 'è', 'é', 'ê', 'î', 'ñ', 'ô', 'ö', 'û', 'ü', 'œ', '—', '‘', '’', '“', '”', '•', '™', '・']
vocab_size = len(vocab)

encode = lambda x: [vocab.index(i) for i in x]
decode = lambda x: ''.join([vocab[i] for i in x])

class trans_block(nn.Module):
    def __init__(self,embed_size,heads):
        super().__init__()
        head_size = embed_size // heads
        self.attention = Heads(heads,head_size)
        self.ff_layer = FF_Layer(embed_size)
        self.lnorm1 = nn.LayerNorm(embed_size)
        self.lnorm2 = nn.LayerNorm(embed_size)
    def forward(self,x):
        x = x + self.attention(self.lnorm1(x))
        x = x + self.ff_layer(self.lnorm2(x))
        return x

class Head(nn.Module):
    def __init__(self,headsize):
        super().__init__()
        self.key = nn.Linear(embed_size,headsize,bias=False)
        self.query = nn.Linear(embed_size,headsize,bias=False)
        self.value = nn.Linear(embed_size,headsize,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        Batches, Time, Channels = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * Channels**-0.5
        wei = wei.masked_fill(self.tril[:Time,:Time] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class Heads(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(n_head)])
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FF_Layer(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size,embed_size),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,embed_size)
        self.position_embedding_table = nn.Embedding(block_size,embed_size)
        self.lm_head = nn.Linear(embed_size,vocab_size)
        self.blocks = nn.Sequential(*[trans_block(embed_size,heads = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_size)
    def forward(self,idx,targets=None):
        Branch,Time = idx.shape

        token_embed = self.embedding_table(idx)
        position_embed = self.position_embedding_table(torch.arange(Time,device=device))
        added = token_embed + position_embed
        added = self.blocks(added)
        added = self.ln_f(added)
        logits = self.lm_head(added)

        if targets is None:
            loss = None
        else:
            Batch, Time, Channel = logits.shape
            logits = logits.view(Batch*Time,Channel)
            targets = targets.view(Batch*Time)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self, idx, max_tokens):
        for i in range(max_tokens):
            idx_condition = idx[:, -block_size:]
            logits, loss = self(idx_condition)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("loading")
model2 = BigramLM()
model2.load_state_dict(torch.load("model.txt",map_location = torch.device(device)))

def generate_text(contextc, tokens):
  print("generating")
  context = torch.tensor([encode(contextc)])
  a = decode(model2.generate(context,max_tokens = tokens)[0].tolist())
  return a

iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Hullo, said the mysterious man standing on the door"),
        gr.Slider(minimum=1, maximum=1000, step=1, label="Number of characters to generate", value=100)
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="HoLLMes",
    description="A janky LLM trained on Detective Novels."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
