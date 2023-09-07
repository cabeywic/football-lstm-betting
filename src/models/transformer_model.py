import torch.nn as nn
import torch


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layer to transform input_dim to d_model as required by transformer
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(1000, d_model)  # Assuming max sequence length is 1000
        
        # Transformer
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # Add positional encoding to the source input
        src = self.embedding(src)
        src = src + self.positional_encoding(torch.arange(src.size(1), device=src.device)).unsqueeze(0)

        # Generate mask for the transformer
        mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Pass through the transformer
        output = self.transformer(src, src, src_mask=mask)
        
        # Extract the output of the last time step
        output = output[:, -1, :]
        
        # Pass through final linear layer
        output = self.fc(output)
        
        return output
