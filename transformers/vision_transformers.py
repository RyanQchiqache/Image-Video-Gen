from cv2 import sqrt
import torch
from torch import nn


class Tokenizer(nn.Module):
    """
        Tokenizer to split the image into fixed sized patches, linearly embed each of them, add position embeddings.
        Output:: (Batch_size, Number_of_Patches, Embedding_dim)

    """
    def __init__(self, in_height, in_width, out_height, out_width, in_channels, emb_dim) -> None:
        super().__init__()


        assert in_height % out_height == 0, f"{in_height} should be devided by {out_height}" 
        assert in_width % out_width == 0, f"{in_width} should be devided by {out_width}"

        kernel_h = in_height // out_height
        kernel_w = in_width // out_width
        kernel_size = kernel_h
        assert kernel_h ==  kernel_w, f"Only square kernels are supported"
        
        self.patch_size = kernel_size

        self.conv = nn.Conv2d(in_channels, emb_dim, kernel_size, stride=kernel_size)
        
        self.flattner = nn.Flatten(2,3)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flattner(x).transpose(-2,-1)
        return x

# Transformer encoder consists of 2 Blocks: (layerNorm + Multihead Attention) & (layerNorm + MLP)

def multi_layer_perceptron(em_dim, mlp_size, dropout):
    return nn.Sequential(
        nn.Linear(em_dim, mlp_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_size, em_dim),
        nn.Dropout(dropout)
    )

def attention(Q,K,V, d_k):
    return nn.Softmax(Q @ K.T / torch.sqrt(d_k)) @ V


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_size, dropout, attention_dropout):
        super().__init__()
        
        self.pre_attention_norm = nn.LayerNorm(emb_dim)
        self.mul_attention = nn.MultiheadAttention(emb_dim, num_heads, attention_dropout)
        self.attention_dropout = nn.Dropout(dropout)

        self.norm_pre_mlp = nn.LayerNorm(emb_dim)
        self.multi_layer_perceptron = multi_layer_perceptron(emb_dim, mlp_size, dropout)


    def forward(self, x):
        # first Blocks
        y = self.pre_attention_norm(x)
        attention_output, attention_weights = self.mul_attention(y, y, y) # 1"output": (B, N:number_of_patches , D:dim/h), 2"weights":(B num_heads, N, N)
        print(type(attention_output), attention_output.shape)
        print(type(attention_weights), attention_weights.shape)
        attention_dropout = self.attention_dropout(attention_output)
        x = x + attention_dropout # residual connection
        # second Block
            
        y = self.norm_pre_mlp(x)
        y = self.multi_layer_perceptron(y)

        x = x + y 
        return x





if __name__=="__main__":
    in_height, in_width, out_height, out_width = 64, 64, 32, 32
    in_channels, embedding_dim = 3, 128 
    fake_batch = torch.rand((1, in_channels, in_height, in_width))
    tokenizer = Tokenizer(in_height, in_width, out_height, out_width, in_channels, embedding_dim )
    tokenizer_result = tokenizer.forward(fake_batch)
    assert tokenizer_result.shape == (1,1024,128), f"something went wrong with your tokenizer"
    print(f"fake batch after tokenizing with fake batch of size: {fake_batch.shape} we have the shape of the output: {tokenizer_result.shape}") 
        # (channels, 32x32 from output of con2d, embedding_dim)
    
    em_dim, mlp_size, att_droput, dropout, num_head = 128, 128, 0.1, 0.1, 2
    transformer_ecoder = TransformerEncoder(em_dim, num_head, mlp_size, dropout, att_droput)
    encoder_result = transformer_ecoder(tokenizer_result)
    print(f"encoder_result.shappe: {encoder_result.shape}")
    
    num_patches = 1024
    embedding = nn.Parameter(torch.empty(num_patches, em_dim))
    _ =  nn.init.trunc_normal_(embedding, std=0.2)
    print(f"positional embeddings: {embedding.std()}, {embedding.shape}, {embedding.min().item()}, {embedding.max().item()}")



    
