####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.nn as nn
from utils.video_encoder import VideoEncoder
from utils.decoder import DecoderLayer, TextEmbedding

class VideoCaptioner(nn.Module):
    def __init__(self, vocab_size, dim=768, num_heads=8, num_layers=4, vis_out_dimension=512, num_frames=16, freeze_vision=True, VisionEncoder=None, max_len=100, vis_hxw_out=49):
        super().__init__()
        assert VisionEncoder != None, "provide a vision encoder class"
        
        self.vision_encoder = VideoEncoder(VisionEncoder(), num_frames, vis_out_dimension, dim, vis_hxw_out)
        self.text_embed = TextEmbedding(vocab_size, dim, max_len=max_len)
        self.decoder_layers = nn.ModuleList([
           DecoderLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(dim, vocab_size)
        if freeze_vision:
           for p in self.vision_encoder.vision_encoder.parameters():
               p.requires_grad = False

    def forward(self, videos, input_ids, return_att=False):
        vision = self.vision_encoder(videos)
        x = self.text_embed(input_ids)
        L = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=input_ids.device), diagonal=1).bool()
        
        att = []
        for layer in self.decoder_layers:
           if return_att:
               x, att = layer(x, vision, causal_mask, return_att=return_att)
           else:
               x = layer(x, vision, causal_mask, return_att=return_att)
        
        ret = (self.lm_head(x), att[-1]) if return_att else self.lm_head(x)
        return ret

@torch.no_grad()
def generate_caption(model, video, voc, max_len=50, return_att=False):
    ret = []
    model.eval()
    video = video.unsqueeze(0)
    start_token_id = voc('<start>')
    end_token_id = voc('<end>')
    device = video.device
    input_ids = torch.tensor([[start_token_id]], device=device)
    
    for _ in range(max_len):
        if return_att:
            logits, att = model(video, input_ids, return_att=return_att)
        else:
            logits = model(video, input_ids)
        
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == end_token_id:
           break
    ret = [voc.idx2word[id] for id in list(input_ids.detach().cpu().numpy()[0])]
    return ret, att if return_att else ret
