import torch
import torch.nn as nn

from transformers import BertModel

from argparse import Namespace

from .config import PRETRAINED_BERT_MAX_TOKENS

class BERTEncoder(nn.Module):
    def __init__(self, args : Namespace):
        """
        A mostly pretrained BERT encoder model, which works with a token count multiplicible by PRETRAINED_BERT_MAX_TOKENS (512)

        Parameters
        ----------
        args : argparse.Namespace
            The program configuration.   
        """
        super(BERTEncoder, self).__init__()

        # number of chunks (C)
        self.n_chunks = args.max_tokens // PRETRAINED_BERT_MAX_TOKENS

        ## Pretrained model that matches the tokenizer
        # https://huggingface.co//google-bert/bert-base-uncased
        # It inherits from torch.nn.Module, so just treat it as anything else.
        self.BERT_encoder = BertModel.from_pretrained("bert-base-uncased")

        # BERT_encoder output on all chunks is shape (B, C, 178)

        ## Bring shape (B, 178, C) to (B, 178, 1)
        # Adaptive pool figures out kernel, stride, padding, etc...
        self.chunk_pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x : torch.Tensor, x_attn_mask : torch.Tensor):
        """
        Encodes tokens into a latent space.

        Parameters
        ----------
        x : torch.Tensor
            The tokenized string split into `C` chunks of size PRETRAINED_BERT_MAX_TOKENS (512). Shape is `(B, C, 512)`.
        x_attn_mask : torch.Tensor
            The attention mask that goes with `x`. Contains the same shape and order as `x`, `(B, C, 512)`.
            
        Returns
        -------
        result : torch.Tensor
            Encoded tensor of shape `(B, 178)`.
        """
        # encoded_chunks = []
        B, C, L = x.shape

        x = x.reshape(B*C, L)
        x_attn_mask = x_attn_mask.reshape(B*C, L)

        encoded_chunks = self.BERT_encoder(input_ids=x, attention_mask=x_attn_mask).pooler_output

        encoded_chunks = encoded_chunks.reshape(B, C, -1)

        return encoded_chunks

class SharedBERT(nn.Module):
    def __init__(self, args):
        super(SharedBERT, self).__init__()

        self.BERT_encoder = BERTEncoder(args)

        # for p in self.BERT_encoder.parameters():
        #     p.requires_grad = False
        # self.BERT_encoder.eval()

        self.similarity = nn.CosineSimilarity(dim=2)

        self.mapping = nn.Linear(self.BERT_encoder.n_chunks,1) # Learn mapping from similarity space to probability

        # self.sigmoid = nn.Sigmoid()

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        resume_encodings = self.BERT_encoder(resume, resume_attn_mask)
        description_encodings = self.BERT_encoder(description, description_attn_mask)


        similarities = self.similarity(resume_encodings, description_encodings)

        # Learn map from [-1, 1] to [0, 1]
        output = self.mapping(similarities)
        # output = self.sigmoid(output)

        return output.flatten()

class TempModel(nn.Module):
    def __init__(self):
        """
        This model is not meant to be used. It's just a placeholder for a model that has not been made yet.
        
        Raises
        ------
        `NotImplementedError` on constructor invocation
        """
        super(TempModel, self).__init__()
        
        raise NotImplementedError("This is a placeholder value, it should not be used.")
    
    def forward(self, x : torch.Tensor):
        pass
