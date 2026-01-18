####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.utils.data as data
import os
import cv2
from PIL import Image

class MsvdDataset(data.Dataset):
    def __init__(self, root, annotation_dict, vocab, transform=None, num_frames=16):
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.num_frames = num_frames
        self.annotations_dict = annotation_dict
    def __getitem__(self, idx):
        video_id = self.annotations_dict[idx]['id']
        caption = " ".join(self.annotations_dict[idx]['caption'])

        video_path = os.path.join(self.root, f'{video_id}.avi')

        frames = self._load_frames(video_path)

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        tokens = self.vocab.custom_word_tokenize(str(caption))
        caption_tensor = torch.Tensor([self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')])

        return frames, caption_tensor

    def __len__(self):
        return len(self.annotations_dict)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = torch.linspace(0, total_frames - 1, self.num_frames, dtype=torch.long)
        
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        
        return frames
