import numpy as np
import torch
import torch.nn as nn


# VQ VAE model
class VQ_VAE(nn.Module):

    def __init__(self, codebook_num=512, codebook_dim=32):
        super().__init__()
        
        self.image_encoder = nn.Sequential(
            #Assuming the image will be a Batch * 256*64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # shape: 32*32*128
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # shape: 64*16*64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # shape: 128*8*32
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # shape: 256*4*16
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten()
        )
        self.encoder = nn.Sequential(
            nn.Linear(256*4*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, codebook_dim),
        )
        self.embeddings = nn.Embedding(codebook_num, codebook_dim)
        self.beta = 0.2
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256*4*16),
        )

        self.image_decoder = nn.Sequential(
            # grid will be b * 256* 4 * 16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )


    def forward(self, x):
        #we will assume the image will be an batch * 62 * 256 image
        b, w, h = x.shape
        x = x.unsqueeze(1)
        x = self.image_encoder(x)
        w_encoded, h_encoded = x.shape
        x_in = self.encoder(x)
        # #Making x shape B*1*dim and embeddings B*10*dim
        dist = torch.cdist(x_in.unsqueeze(1), torch.tile(self.embeddings.weight.unsqueeze(0), (b, 1, 1)))
        
        #Get the smallest index from codebook for each batch: B*1
        indices = torch.argmin(dist, dim=2).flatten()
        x_out = self.embeddings(indices).squeeze(1)

        #Computing the Losses
        commitementL = nn.functional.mse_loss(x_out.detach(), x_in, reduction="mean")
        codebookL = nn.functional.mse_loss(x_out, x_in.detach(), reduction="mean")
        quantizationL = self.beta * commitementL + codebookL
        #Straight Through Estimation
        x_out = x_in + (x_out - x_in).detach()

        #Decoder
        x = self.decoder(x_out)
        x = x.reshape(b, 256, 4, 16)
        x = self.image_decoder(x)
        out = x.reshape(b, w, h)

        return out, quantizationL, indices


class VQ_VAE_CNN(nn.Module):
    def __init__(self, codebook_num=512, codebook_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 128x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x8
            nn.ReLU(),
            nn.Conv2d(128, codebook_dim, 4, stride=2, padding=1)  # 16x4xembedding_dim
        )
        
        # Vector Quantization
        self.embedding = nn.Embedding(codebook_num, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_num, 1.0 / codebook_num)
        self.commitment_cost = 0.25
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(codebook_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def vector_quantize(self, z):
        # z shape: [B, D, H, W]
        b, d, h, w = z.shape
        
        # Flatten z
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, d)  # [BHW, D]
        
        # Calculate distances
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # Find nearest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.weight.shape[0]).to(z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(b, h, w, d)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Loss
        q_latent_loss = torch.mean((quantized.detach() - z)**2)
        e_latent_loss = torch.mean((quantized - z.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = z + (quantized - z).detach()
        return quantized, loss, encoding_indices.view(b, h, w)
 
    def forward(self, x):
        b, w, h = x.shape
        x = x.unsqueeze(1)
        x = self.encoder(x)
        q, qloss, indices = self.vector_quantize(x)
        out = self.decoder(q)
        return out.unsqueeze(1), qloss, indices












