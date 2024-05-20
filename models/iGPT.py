import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

n_head = 4
n_layer = 2
dropout = 0.2
n_embd=128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Head(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # dropout
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class AutoregressiveTransformer(nn.Module): 
    def __init__(self, input_dim=3, n_head=4,
                 num_layers=2,
                 max_seq_length=401):
        


        """
        Autoregressive Transformer for image generation.
        
        :param input_dim: The size of the input dimension (number of pixel values + 1 for <bos> token).
        :param d_model: The dimension of the embeddings and transformer hidden layers.
        :param nhead: The number of heads in the multi-head attention models.
        :param num_encoder_layers: The number of layers in the transformer encoder.
        :param dim_feedforward: The dimension of the feedforward network in the transformer.
        :param max_seq_length: The maximum length of the sequences (flattened image size).
        """



        super().__init__()

        # Embedding layer to convert input tokens to dense vectors
        self.token_embedding_table = nn.Embedding(input_dim, n_embd)

        # Positional encoding to add sequence information to embeddings
        self.position_embedding_table = nn.Embedding(max_seq_length, n_embd)

        # Custom Transformer Decoder Layer with GeLU activation
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=max_seq_length) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, input_dim)


    def forward(self, idx):
        # src shape: [batch_size, seq_len]

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits
    def map_rgb_to_token(rgb_pixel):
        # Assuming rgb_pixel is in the format [R, G, B] with 2-bit per channel
        token = (rgb_pixel[0] << 4) + (rgb_pixel[1] << 2) + rgb_pixel[2]
        
        return token
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def preprocess_data(self, data, bos_token=2):
        """
        Preprocess color image data for the autoregressive transformer.
        Each RGB pixel is mapped to a single token, and a <bos> token is added at the beginning.

        :param data: A (n_samples, H, W, C) numpy array of images with 2-bit per channel.
        :param bos_token: An integer representing the <bos> token.
        :return: A PyTorch tensor of shape (n_samples, H*W + 1).
        """
        n_samples, H, W, C = data.shape

        # Function to map each RGB pixel to a cluster token

        # Apply the mapping to each pixel
        data_tokenized = np.apply_along_axis(self.map_rgb_to_token, -1, data)

        # Flatten the image
        data_tokenized = data_tokenized.reshape(n_samples, -1)  # Shape: (n_samples, H*W)
        data_tokenized = torch.tensor(data_tokenized, dtype=torch.long)

        # Add <bos> token to the beginning of each image
        bos_tokens = torch.full((n_samples, 1), bos_token, dtype=torch.long)
        data_tokenized = torch.cat([bos_tokens, data_tokenized], dim=1)

        return data_tokenized

    def generate_samples(self, num_samples, image_shape, device, bos_token=2):
        H, W, _ = image_shape  # C is not used directly as each token represents a full RGB pixel
        src = torch.full((num_samples, 1), bos_token, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(H * W):  # Loop for H*W instead of H*W*C
                logits = self.forward(src)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                src = torch.cat((src, idx_next), dim=1)

        # Function to map each token back to RGB values
        def map_token_to_rgb(token):
            r = (token >> 4) & 0x03
            g = (token >> 2) & 0x03
            b = token & 0x03
            return [r, g, b]

        # Convert tokens to RGB values and reshape
        generated_images = src[:, 1:].cpu().numpy()
        generated_images_rgb = np.array([[map_token_to_rgb(token) for token in row] for row in generated_images])
        generated_images_rgb = generated_images_rgb.reshape(num_samples, H, W, 3)  # Reshape to (num_samples, H, W, C)

        return generated_images_rgb
    def loss(self, x):
        criterion = nn.CrossEntropyLoss()        
        train_processed = self.preprocess_data(x)
        inputs = train_processed[:,:-1].to(device)
        targets = train_processed[:,1:].to(device)
        outputs = self.forward(inputs)
        return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

    def training(self, x):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        losses = self.loss(x)
        losses.backward()
        optimizer.step()
        
        


def q3_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets
    
    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """ 
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10

    # Model Initialization
    model = AutoregressiveTransformer(input_dim=65, max_seq_length=(image_shape[0]*image_shape[1] + 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    # Data Preprocessing
    # - Flatten images
    # - Add <bos> tokens
    train_data_processed = model.preprocess_data(train_data)
    test_data_processed = model.preprocess_data(test_data)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        np.random.shuffle(train_data)
        # Calculate the number of batches
        num_batches = train_data.shape[0] // batch_size
    
        for i in range(train_data.shape[0] // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = train_data_processed[start_idx:end_idx]

            inputs = batch_data[:, :-1].to(device)  # All tokens except the last
            targets = batch_data[:, 1:].to(device)  # All tokens except the first
    
        
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_train_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Learning rate scheduling
            scheduler.step()

            train_losses.append(loss.item())

        # Evaluate on the test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            num_testbatch = test_data.shape[0] // batch_size
            for i in range(test_data.shape[0] // batch_size):
                batch_data = test_data_processed[i * batch_size:(i + 1) * batch_size]
                inputs = batch_data[:, :-1].to(device)  # All tokens except the last
                targets = batch_data[:, 1:].to(device)  # All tokens except the first
        
                outputs = model(inputs)
                test_loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_test_loss += test_loss.item()
                test_losses.append(test_loss.item())

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss/num_batches:.4f}, Test Loss: {total_test_loss/num_testbatch:.4f} lr: {scheduler.get_last_lr()[0]}')


    # Sampling new images
    samples = model.generate_samples(num_samples=100, image_shape=image_shape, device=device)