import attr
import torch
import torch.nn as nn
import numpy as np

from . import net_utils
import utils

@attr.s
class EncoderOutput:
    q_enc = attr.ib()
    q_len = attr.ib()
    col_enc = attr.ib()
    col_name_len = attr.ib()
    col_len = attr.ib()


class BaselineEncoder(nn.Module):
    def __init__(self, N_word, N_h, N_depth, word_embedding_layer, encode_cols):
        super(BaselineEncoder, self).__init__()
        self.N_word = N_word
        self.N_h = N_h
        self.N_depth = N_depth

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.word_embedding_layer = word_embedding_layer
        self.encode_cols = encode_cols

        self.zero_emb = np.zeros(self.N_word, dtype=np.float32)
        
        if encode_cols:
            self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                    num_layers=N_depth, batch_first=True,
                    dropout=0.3, bidirectional=True)

    def forward(self, data, perm, st, ed, table_type):
        # Lookup embeddings for questions
        q_seq = []
        for permuted_idx in perm[st:ed]:
            q_seq.append(data[permuted_idx]['question_tokens'])
        q_emb_var, q_len = self.word_embedding_layer.gen_x_q_batch(q_seq)

        # Run question through bidirectional LSTM
        q_enc, _ = net_utils.run_lstm(self.q_lstm, q_emb_var, q_len)

        if self.encode_cols:
            col_seq = utils.to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = self.word_embedding_layer.gen_col_batch(col_seq)
            col_enc, _ = net_utils.col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)
        else:
            col_enc = None
            col_name_len = None
            col_len = None
        return EncoderOutput(q_enc, q_len, col_enc, col_name_len, col_len)