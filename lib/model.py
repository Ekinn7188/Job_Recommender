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
        self.BERT_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased")

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
        encoded_chunks = []

        # iterate over chunks and encode them piece-by-piece
        for i in range(x.shape[1]):
            loop_x = x[:, i, :]
            loop_attn_mask = x_attn_mask[:, i, :]

            # NOTE: Type of output from self.BERT_encoder() is transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
            encoded_chunk : torch.Tensor = self.BERT_encoder(input_ids=loop_x, attention_mask=loop_attn_mask).pooler_output

            ## restore chunk dim
            # Shape is (B, 1, 178)
            encoded_chunk = encoded_chunk.unsqueeze(1)

            encoded_chunks.append(encoded_chunk)

        # join back together
        encoded_chunks = torch.cat(encoded_chunks, dim=1)

        encoded_chunks = encoded_chunks.permute(0,2,1) # swap dim 1 and 2

        # pool chunks
        result = self.chunk_pooling(encoded_chunks) # shape is (B, 178, 1)

        result = result.squeeze(2) # shape is (B, 178)

        return result

class SharedBERT(nn.Module):
    def __init__(self, args):
        super(SharedBERT, self).__init__()

        self.BERT_encoder = BERTEncoder(args)

        self.similarity = nn.CosineSimilarity(dim=1)

        self.mapping = nn.Linear(1,1) # Learn mapping from similarity space to probability

        self.sigmoid = nn.Sigmoid()

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        resume_encoding = self.BERT_encoder(resume, resume_attn_mask)
        description_encoding = self.BERT_encoder(description, description_attn_mask)

        output = self.similarity(resume_encoding, description_encoding)

        output = output.unsqueeze(1) # bring from (B,) to (B, 1)

        # Learn map from [-1, 1] to [0, 1]
        output = self.mapping(output)
        output = self.sigmoid(output)

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
