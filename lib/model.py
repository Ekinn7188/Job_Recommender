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

        encoded_chunks = self.BERT_encoder(input_ids=x, attention_mask=x_attn_mask).last_hidden_state #[:,0,:] #.pooler_output

        encoded_chunks = encoded_chunks.reshape(B, C*L, -1)

        return encoded_chunks

class SharedBERT(nn.Module):
    def __init__(self, args):
        super(SharedBERT, self).__init__()

        self.BERT_encoder = BERTEncoder(args)

        # for p in self.BERT_encoder.parameters():
        #     p.requires_grad = False
        # self.BERT_encoder.eval()

        self.similarity = nn.CosineSimilarity(dim=1)

        self.mapping = nn.Linear(1,3) # Learn mapping from similarity space to class probabiltiies

        self.softmax = nn.Softmax(dim=1)

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        resume_encodings = self.BERT_encoder(resume, resume_attn_mask).mean(dim=1)
        description_encodings = self.BERT_encoder(description, description_attn_mask).mean(dim=1)

        similarities = self.similarity(resume_encodings, description_encodings)
        similarities = similarities.unsqueeze(1)
        # Learn map from [-1, 1] to [0, 1]
        output = self.mapping(similarities)
        output = self.softmax(output)

        return output

class SplitBERT(nn.Module):
    def __init__(self, args):
        super(SplitBERT, self).__init__()

        self.resume_encoder = BERTEncoder(args)
        self.description_encoder = BERTEncoder(args)

        self.hidden_size = 768
        self.n_heads = 4
        self.dropout = 0.1

        self.cross_attention_r2d = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_heads,
            batch_first=True,
            dropout=self.dropout
        )
        
        self.cross_attention_d2r = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_heads,
            batch_first=True,
            dropout=self.dropout
        )

        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size),
            nn.Linear(self.hidden_size,self.hidden_size//2),
            nn.Linear(self.hidden_size//2, 3),
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        B, C, L = resume_attn_mask.shape

        resume_encodings = self.resume_encoder(resume, resume_attn_mask)
        resume_attn_mask = resume_attn_mask.reshape(B, C*L)

        description_encodings = self.description_encoder(description, description_attn_mask)
        description_attn_mask = description_attn_mask.reshape(B, C*L)

        ## cross attention

        # query the description with the resume
        r_context, _ = self.cross_attention_r2d(
            query=resume_encodings,
            key=description_encodings,
            value=description_encodings,
            key_padding_mask=(resume_attn_mask == 0), # padding is True for pad, False for no pad,
            need_weights=False
        )

        # query the resume with the description 
        d_context, _ = self.cross_attention_d2r(
            query=description_encodings,
            key=resume_encodings,
            value=resume_encodings,
            key_padding_mask=(description_attn_mask == 0), # padding is True for pad, False for no pad,
            need_weights=False
        )

        resume_encodings = resume_encodings + r_context
        description_encodings = description_encodings + d_context

        # Head
        resume_encodings = self.conv(resume_encodings.transpose(1,2)).transpose(1,2) # swap sequence and hidden, then put back
        description_encodings = self.conv(description_encodings.transpose(1,2)).transpose(1,2) # swap sequence and hidden, then put back

        resume_encodings = self.masked_mean(resume_encodings, resume_attn_mask)
        description_encodings = self.masked_mean(description_encodings, description_attn_mask)

        output = torch.concat([resume_encodings, description_encodings], dim=1)

        output = self.linear(output)
        output = self.softmax(output)

        return output

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1)

        return x.sum(dim=1)/lengths


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
