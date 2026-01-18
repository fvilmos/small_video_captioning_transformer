####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import re
import json

class Vocabulary:
    """
    Generate a vocabulary of works -> tokens.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
        
    def __len__(self):
        return len(self.word2idx)

    def export_vocabulary(self, voc_name):
        with open(voc_name, 'w') as f:
            json.dump(self.idx2word, f, indent=4)

    def import_vocabulary(self, voc_name):
        with open(voc_name, 'r') as f:
            t_idx2word = json.load(f)
        # format the index
        self.idx2word = {int(k):v for k,v in t_idx2word.items()}

        # invert k,v get word2idx variable
        self.word2idx = {v:int(k) for k,v in t_idx2word.items()}

    def custom_word_tokenize(self, text):
        text = text.lower()
        # split by spaces and, keep punctuation as separate tokens
        tokens = re.findall(r"[\w']+|[.,!?;()]", text)
        return tokens