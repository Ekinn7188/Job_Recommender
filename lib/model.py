import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

from argparse import Namespace

from .config import PRETRAINED_BERT_MAX_TOKENS

import os
import polars as pl
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

from gensim.models import Word2Vec 

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
            Encoded, unraveled, chunk tensor of shape `(B, C*512, 178)`.
        """
        B, C, L = x.shape

        x = x.reshape(B*C, L)
        x_attn_mask = x_attn_mask.reshape(B*C, L)

        encoded_chunks = self.BERT_encoder(input_ids=x, attention_mask=x_attn_mask).last_hidden_state

        encoded_chunks = encoded_chunks.reshape(B, C, L, -1)

        return encoded_chunks

class TypeClassifierBERT(nn.Module):
    def __init__(self, args):
        super(TypeClassifierBERT, self).__init__()

        self.BERT_encoder = BERTEncoder(args)

        self.hidden_size = 768 # output from BERT

        self.num_classes = 43 # from the type classification dataset

        self.dropout = 0.1

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2,self.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//4, self.num_classes),
        )

    def forward(self, x, x_attn_mask):
        x = self.BERT_encoder(x, x_attn_mask)

        x = x[:, :, 0, :] # get [CLS] from every chunk. Shape is (B, C, H)

        x_attn_mask = x_attn_mask.any(dim=2).float()

        x = self.masked_mean(x, x_attn_mask) # Shape is (B, H)

        x = self.linear(x)

        return x




    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1)

        return x.sum(dim=1)/lengths

class SharedBERT(nn.Module):
    def __init__(self, args):
        super(SharedBERT, self).__init__()

        self.BERT_encoder = BERTEncoder(args)

        # for p in self.BERT_encoder.parameters():
        #     p.requires_grad = False
        # self.BERT_encoder.eval()

        self.hidden_size = 768
        
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size*4,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 3),
        )

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        B, C, L = resume_attn_mask.shape

        resume_encodings = self.BERT_encoder(resume, resume_attn_mask)[:, :, 0, :] # get CLS
        description_encodings = self.BERT_encoder(description, description_attn_mask)[:, :, 0, :]  # get CLS

        chunk_mask_r = resume_attn_mask.any(dim=2).float()
        chunk_mask_d = description_attn_mask.any(dim=2).float()

        resume_encodings = self.masked_mean(resume_encodings, chunk_mask_r)
        description_encodings = self.masked_mean(description_encodings, chunk_mask_d)

        output = torch.cat([resume_encodings, 
                            description_encodings, 
                            torch.abs(resume_encodings - description_encodings), 
                            resume_encodings * description_encodings], dim=1)
        
        # print(output.shape)

        output = self.linear(output)

        return output

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1)

        return x.sum(dim=1)/lengths
    
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
            nn.Linear(self.hidden_size*4,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 3),
        )

    def forward(self, resume, resume_attn_mask, description, description_attn_mask):
        B, C, L = resume_attn_mask.shape

        resume_encodings = self.resume_encoder(resume, resume_attn_mask)
        resume_encodings = resume_encodings.reshape(B, C*L, -1)
        resume_attn_mask = resume_attn_mask.reshape(B, C*L)

        description_encodings = self.description_encoder(description, description_attn_mask)
        description_encodings = description_encodings.reshape(B, C*L, -1)
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

        resume_encodings = (resume_encodings + r_context)
        description_encodings = (description_encodings + d_context)

        resume_encodings = resume_encodings.reshape(B, C, L, -1)[:, :, 0, :] # get CLS
        resume_attn_mask = resume_attn_mask.reshape(B, C, L)

        description_encodings = description_encodings.reshape(B, C, L, -1)[:, :, 0, :] # get CLS
        description_attn_mask = description_attn_mask.reshape(B, C, L)

        # Head

        chunk_mask_r = resume_attn_mask.any(dim=2).float()
        chunk_mask_d = description_attn_mask.any(dim=2).float()

        resume_encodings = self.masked_mean(resume_encodings, chunk_mask_r)
        description_encodings = self.masked_mean(description_encodings, chunk_mask_d)

        output = torch.cat([resume_encodings, 
                            description_encodings, 
                            torch.abs(resume_encodings - description_encodings), 
                            resume_encodings * description_encodings], dim=1)

        output = self.linear(output)

        return output

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1)

        return x.sum(dim=1)/lengths
    
class Word2VecLSTM(nn.Module):
    def __init__(self, args):
        super(Word2VecLSTM, self).__init__()

        self.args = args
        self.embedding_dim = 100 # keeping it light so <300
        self.hidden_size = 128

        # path to save/load the trained Word2Vec model
        self.w2v_path = os.path.join(args.dataset_dir, "w2v.model")

        # build or load Word2Vec
        if os.path.exists(self.w2v_path):
            self.w2v = Word2Vec.load(self.w2v_path)
        else:
            raise RuntimeError(
                "w2v.model not found. Train Word2Vec once on your CPU node and save it "
                "to dataset/w2v.model before running the LSTM model."
            )

        # build vocab index from gensim model
        vocab = list(self.w2v.wv.index_to_key)
        self.word2idx = {w: i+1 for i, w in enumerate(vocab)} # 0 reserved for PAD
        self.word2idx["<PAD>"] = 0

        # build embedding weight matrix
        weights = torch.zeros((len(vocab) + 1, self.embedding_dim))
        for i, w in enumerate(vocab):
            weights[i+1] = torch.tensor(self.w2v.wv[w])

        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # interaction head (same pattern as BERT head)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def encode(self, x, mask):
        """
        x: (B, L) indices
        mask: (B, L) 1 for real tokens, 0 for padding
        """
        emb = self.embedding(x) # (B, L, E)
        # LSTM over full sequence.
        _, (h, _) = self.lstm(emb) # h: (1, B, H)
        h = h.squeeze(0) # (B, H)
        return h

    def forward(self, resume, resume_mask, description, description_mask):
        res_vec = self.encode(resume, resume_mask)
        job_vec = self.encode(description, description_mask)

        x = torch.cat(
            [
                res_vec,
                job_vec,
                torch.abs(res_vec - job_vec),
                res_vec * job_vec,
            ],
            dim=1,
        )

        return self.head(x)
    
class TFIDFLogReg:
    def __init__(self, args):
        self.args = args
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1
        )
        # 3 fits: none, potential, good
        self.label_map = {
            "No Fit": 0,
            "Potential Fit": 1,
            "Good Fit": 2
        }
        
    def load_split(self, name):
        df = pl.read_csv(os.path.join(self.args.dataset_dir, f"{name}.csv"))
        texts = (
            df["resume_text"].cast(str)
            + " [SEP] "
            + df["job_description_text"].cast(str)
        ).to_list()
        
        labels = df["label"].map_elements(lambda x: self.label_map[x]).to_numpy()
        
        return texts, labels
    
    def train(self):
        X_train, y_train = self.load_split("train")
        
        # shuffle before splitting
        indices = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in indices]
        y_train = y_train[indices]
        
        n_train = int(0.9 * len(X_train))

        X_t = X_train[:n_train]
        X_val = X_train[n_train:]
        y_t = y_train[:n_train]
        y_val = y_train[n_train:]

        X_t = self.vectorizer.fit_transform(X_t)
        X_val = self.vectorizer.transform(X_val)

        self.model.fit(X_t, y_t)

        # evaluate
        y_prob = self.model.predict_proba(X_val)
        y_pred = y_prob.argmax(axis=1)

        return {
            "val_acc": float(accuracy_score(y_val, y_pred)),
            "val_prec": float(precision_score(y_val, y_pred, average="macro")),
            "val_rec": float(recall_score(y_val, y_pred, average="macro")),
            "val_ce": float(log_loss(y_val, y_prob, labels=[0,1,2]))
        }

    def test(self):
        X_test, y_test = self.load_split("test")
        X_test = self.vectorizer.transform(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
        y_prob = self.model.predict_proba(X_test)
        y_pred = y_prob.argmax(axis=1)

        return {
            "test_acc": float(accuracy_score(y_test, y_pred)),
            "test_prec": float(precision_score(y_test, y_pred, average="macro")),
            "test_rec": float(recall_score(y_test, y_pred, average="macro")),
            "test_ce": float(log_loss(y_test, y_prob, labels=[0,1,2]))
        }

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
