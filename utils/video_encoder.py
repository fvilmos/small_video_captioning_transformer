####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, vit_b_32, ViT_B_32_Weights

class MobileNetV2Encoder(torch.nn.Module):
    """
    Use MobileNetV2 as backbone
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        
        # pretrained model, get the features
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # fetures 1280, 7, 7
        self.features = mobilenet.features
        
    def forward(self, x):
        x = self.features(x) # in: B, C, H, W
        x = x.flatten(2).permute(0, 2, 1) # out: B, HxW, C
        return x

class CNNEncoder(nn.Module):
    """
    Use CNN as backbone
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        
        cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # B,C,H,W = 512,7,7, get just the features
        net = torch.nn.Sequential(*list(cnn.children())[:-2])
        self.net = net
        
    def forward(self, x):
        x = self.net(x).flatten(2).permute(0, 2, 1)
        # B, HxW, D
        return x

class ViTEncoder(nn.Module):
    """
    Use ViT as backbone
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        
        vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        
        self.vit = vit
       
    def forward(self, x):
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)
        # (B, N+1, D)
        return x

class TemporalEmbedding(nn.Module):
    """
    Used to provide ositional information for the frames
    """
    def __init__(self, num_frames, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_frames, embedding_dim)

    def forward(self, x):
        # x size: b,f,hxw,d
        # generate for n frames
        positions = torch.arange(0, x.size(1), device=x.device).long()

        # f,d
        t_emb = self.embedding(positions)

        # reshape to [1,f,1,d]
        t_emb = t_emb.unsqueeze(0).unsqueeze(2)

        x = x + t_emb
        return x

class VideoEncoder(nn.Module):
    """
    Video Encoder that extends the Image Encoder with temporal convolution
    """
    def __init__(self, vision_encoder, num_frames, in_dim, out_dim, vis_hxw_out):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.temporal_embedding = TemporalEmbedding(num_frames, in_dim)
        self.temporal_conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim) # depthwise
        self.projection = VisionProjection(in_dim, out_dim, num_frames * vis_hxw_out)
        self.in_dim = in_dim

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w) # out: bxf,c,h,w
        x = self.vision_encoder(x) # out: bxf,hxw,d
        
        x = x.view(batch_size, num_frames, -1, self.in_dim) # out: b,f,hxw,d
        x = self.temporal_embedding(x) # out: b,f,hxw,d
        
        x = torch.reshape(x,[batch_size,-1,self.in_dim]) # out: b,fxhxw,d

        x = x.permute(0, 2, 1) # out: b,d,fxhxw
        x = self.temporal_conv(x) # out: b,d,fxhxw
        
        x = x.permute(0, 2, 1) # out: b,fxhxw,d
        x = self.projection(x)
        
        return x

class VisionProjection(nn.Module):
    """
    Projects Image Encoder dimensions to the text embedding dimensions
    """
    def __init__(self, in_dim, out_dim, vis_hxw_out):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # add positional encoding as learnable param
        self.pos_encoder = nn.Parameter(torch.randn(1, vis_hxw_out, out_dim))

    def forward(self, x):
        x = self.norm(self.proj(x))
        x = x + self.pos_encoder
        return x
