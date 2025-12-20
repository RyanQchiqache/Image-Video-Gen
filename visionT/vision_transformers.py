import torch
from torch import nn



class Tokenizer(nn.Module):
    """
    Tokenizer to split the image into fixed sized patches, linearly embed each of them.
    Output: (Batch_size, Number_of_Patches, Embedding_dim)
    """

    def __init__(self, in_height, in_width, out_height, out_width, in_channels, emb_dim) -> None:
        super().__init__()

        # out_height, out_width describe the number of patches along H and W
        assert in_height % out_height == 0, f"{in_height} should be divided by {out_height}"
        assert in_width % out_width == 0, f"{in_width} should be divided by {out_width}"

        kernel_h = in_height // out_height
        kernel_w = in_width // out_width
        kernel_size = kernel_h
        assert kernel_h == kernel_w, "Only square kernels are supported"

        self.patch_size = kernel_size
        self.out_height = out_height
        self.out_width = out_width
        self.num_patches = out_height * out_width

        self.conv = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=kernel_size,
            stride=self.patch_size,
        )

        self.flattner = nn.Flatten(2, 3)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)                  # (B, emb_dim, out_height, out_width)
        x = self.flattner(x)              # (B, emb_dim, num_patches)
        x = x.transpose(1, 2)             # (B, num_patches, emb_dim)
        return x

    def positional_embedding(self, num_patches, emb_dim):
        # You *can* ignore num_patches and use self.num_patches, but I keep your API
        embedding = nn.Parameter(torch.empty(1, num_patches, emb_dim))
        return embedding


# Transformer encoder consists of 2 Blocks: (layerNorm + Multihead Attention) & (layerNorm + MLP)



class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_size, dropout, attention_dropout):
        super().__init__()
        
        self.pre_attention_norm = nn.LayerNorm(emb_dim)
        self.mul_attention = nn.MultiheadAttention(
            emb_dim, num_heads, attention_dropout, batch_first=True
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.norm_pre_mlp = nn.LayerNorm(emb_dim)
        self.multi_layer_perceptron_a = self.multi_layer_perceptron(emb_dim, mlp_size, dropout)


    def forward(self, x):
        # first Blocks
        y = self.pre_attention_norm(x)
        attention_output, _  = self.mul_attention(y, y, y) # 1"output": (B, N:number_of_patches , D:dim/h), 2"weights":(B num_heads, N, N)
        attention_dropout = self.attention_dropout(attention_output)
        x = x + attention_dropout # residual connection
        # second Block
            
        y = self.norm_pre_mlp(x)
        y = self.multi_layer_perceptron_a(y)
        x = x + y 

        return x

    def multi_layer_perceptron(self, em_dim, mlp_size, dropout):
        return nn.Sequential(
            nn.Linear(em_dim, mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, em_dim),
            nn.Dropout(dropout)
        )



class ViT(nn.Module):
    def __init__(self, in_height, in_width, out_height, out_width, in_channels, emb_dim, num_heads, mlp_size, dropout, att_dropout, num_classes, num_layers) -> None:
        super().__init__()

        # construct the Tokenizer
        self.tokenizer = Tokenizer(in_height=in_height, in_width=in_width, out_height=out_height, out_width=out_width, in_channels=in_channels, emb_dim=emb_dim)
        
        # construct the positional_embedding
        num_patches = out_height * out_width 
        # construct the class token 
        self.class_token = nn.Parameter(torch.empty(1, 1,  emb_dim)) 
        nn.init.trunc_normal_(self.class_token, std=0.2)    


        # construct the positional_embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1 , emb_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.transformer_encoders = nn.Sequential(
            *[
                TransformerEncoder(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    dropout=dropout,
                    attention_dropout=att_dropout
                ) 
                for _ in range(num_layers)
            ]
        ) 
        
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(att_dropout)
        self.classification = nn.Linear(emb_dim, num_classes)


    def forward(self, x):
        # x: (B, C, H, W)
        patch_embeding = self.tokenizer(x) # (B, N, D)
        
       
        B = patch_embeding.size(0)
        
        cls_token = self.class_token.expand(B, -1, -1) # (B, 1, D)

        x = torch.cat([cls_token, patch_embeding], dim=1) # (B, N+1, D)

        # adding positional embedding
        x = x + self.pos_embedding

        x = self.dropout(x)
        
        for block in self.transformer_encoders:
            x = block(x)

        x = self.layer_norm(x)
        
        # taking CLS token
        Z_out = x[:,0] # (B, D) 

        return self.classification(Z_out) # (B, num_classes)





    
if __name__ == "__main__":
    # ------------------------------
    # 1. Test Tokenizer
    # ------------------------------
    in_height, in_width = 64, 64
    out_height, out_width = 32, 32
    in_channels, emb_dim = 3, 128

    fake_batch = torch.rand((1, in_channels, in_height, in_width))

    tokenizer = Tokenizer(in_height, in_width, out_height, out_width, in_channels, emb_dim)
    tokens = tokenizer(fake_batch)

    expected_patches = out_height * out_width
    assert tokens.shape == (1, expected_patches, emb_dim), \
        f"Tokenizer output incorrect, got {tokens.shape}, expected (1,{expected_patches},{emb_dim})"

    print("Tokenizer OK:", tokens.shape)

    # ------------------------------
    # 2. Test Transformer Encoder
    # ------------------------------
    encoder = TransformerEncoder(emb_dim=emb_dim, num_heads=8, mlp_size=128, dropout=0.1, attention_dropout=0.1)
    encoded = encoder(tokens)

    assert encoded.shape == tokens.shape, \
        f"TransformerEncoder must preserve shape, got {encoded.shape} but expected {tokens.shape}"

    print("TransformerEncoder OK:", encoded.shape)

    # ------------------------------
    # 3. Test Positional Embeddings
    # ------------------------------
    vit_num_patches = expected_patches + 1  # +1 for CLS
    vit_pos = nn.Parameter(torch.zeros(1, vit_num_patches, emb_dim))

    assert vit_pos.shape == (1, vit_num_patches, emb_dim), \
        f"Positional embedding wrong shape: {vit_pos.shape}"

    print("Positional Embedding OK:", vit_pos.shape)

    # ------------------------------
    # 4. Test Full ViT Model
    # ------------------------------
    num_layers = 4
    num_classes = 1000
    num_heads = 8
    mlp_size = 128
    dropout = 0.1
    att_dropout = 0.1

    vit = ViT(
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        in_channels=in_channels,
        emb_dim=emb_dim,
        num_heads=num_heads,
        mlp_size=mlp_size,
        dropout=dropout,
        att_dropout=att_dropout,
        num_classes=num_classes,
        num_layers=num_layers
    )

    vit_output = vit(fake_batch)

    assert vit_output.shape == (1, num_classes), \
        f"ViT output incorrect: expected (1,{num_classes}), got {vit_output.shape}"

    print("ViT Full Model OK:", vit_output.shape)



    
