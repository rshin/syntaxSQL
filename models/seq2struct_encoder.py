import attr
import numpy as np
import torch
import torch.nn as nn

from seq2struct.models import spider_enc
from seq2struct.utils import batched_sequence
from seq2struct.utils import registry


@attr.s
class EncoderOutput:
    q_enc = attr.ib()
    q_len = attr.ib()
    col_enc = attr.ib()
    col_name_len = attr.ib()
    col_len = attr.ib()


class FakePreproc:
    vocab = None


class Seq2structEncoder(nn.Module):
    def __init__(self, N_word, N_h, N_depth, word_embedding_layer, encode_cols, spider_enc_config, gpu):
        super(Seq2structEncoder, self).__init__()
        self.N_word = N_word
        self.N_h = N_h
        self.N_depth = N_depth

        self.word_embedding_layer = word_embedding_layer
        self.encode_cols = encode_cols
        self.zero_emb = np.zeros(self.N_word, dtype=np.float32)
        self.gpu = gpu

        self.spider_enc = registry.instantiate(
            spider_enc.SpiderEncoderV2,
            spider_enc_config,
            device=torch.device('cuda') if gpu else torch.device('cpu'),
            preproc=FakePreproc,
            word_emb_size=N_word,
            recurrent_size=N_h,
        )

    
    def _lookup_embeddings(self, seqs):
        # shape: [sum of seq lengths, 1, N_word]
        embs = torch.from_numpy(np.stack([
            self.word_embedding_layer.word_emb.get(token, self.zero_emb)
            for seq in seqs
            for token in seq
        ], axis=0)).unsqueeze(1)
        if self.gpu:
            embs = embs.cuda()
        boundaries = np.cumsum([0] + [len(seq) for seq in seqs])

        return embs, boundaries

    def _lookup_embeddings_batched(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token 
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.N_word,),
            tensor_type=torch.FloatTensor,
            item_to_tensor=lambda token, batch_idx, out: out.copy_(torch.from_numpy(
                self.word_embedding_layer.word_emb.get(token, self.zero_emb))),
        )
        if self.gpu:
          all_embs = all_embs.apply(lambda d: d.cuda())

        # boundaries shape: [batch, num descs + 1]
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists]

        return all_embs, boundaries

    def _pad_sequences(self, seq_encs):
        # each element of seq_encs has shape
        # [1, seq length, emb size]
        # returns [batch size, max length, emb size]
        max_length = max(seq_enc.shape[1] for seq_enc in seq_encs)
        result = seq_encs[0].data.new(len(seq_encs), max_length, *seq_encs[0].shape[2:]).fill_(0)
        for i, seq_enc in enumerate(seq_encs):
            result[i, :seq_enc.shape[0]] = seq_enc[:, 0]
        return result

    def forward_unbatched(self, data, perm, st, ed, table_type):
        batch_size = ed - st

        q_encs = []
        col_encs = []
        # Number of columns in each entry of batch
        col_lens = []

        # TODO need to call forward_unbatched versions for this to work
        for permuted_idx in perm[st:ed]:
            q_enc, (_, _) = self.spider_enc.question_encoder(self._lookup_embeddings([data[permuted_idx]['question_tokens']]))

            table_names, column_names, column_types, \
                column_to_table, table_to_column, foreign_keys, \
                foreign_keys_tables, primary_keys = data[permuted_idx]['ts']
            desc = {
                'columns': [[column_type] + column_name.split() for (table_id, column_name), column_type in zip(column_names, column_types)],
                'tables': [table_name.split() for table_name in table_names],
                'column_to_table': column_to_table,
                'table_to_column': table_to_column,
                'foreign_keys': foreign_keys,
                'foreign_keys_tables': foreign_keys_tables,
                'primary_keys': primary_keys,
            }

            c_enc, c_boundaries = self.spider_enc.column_encoder(
                self._lookup_embeddings(desc['columns']))
            t_enc, t_boundaries = self.spider_enc.table_encoder(
                self._lookup_embeddings(desc['tables']))
            #assert np.all((c_boundaries[1:] - c_boundaries[:-1]) == 1)
            
            q_enc_new, c_enc_new, t_enc_new = self.spider_enc.encs_update(
                desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)
            
            q_encs.append(q_enc_new)
            col_encs.append(c_enc_new)
            col_lens.append(len(desc['columns']))
        
        q_enc = self._pad_sequences(q_encs)
        q_len = np.array([q_enc_elem.shape[1] for q_enc_elem in q_encs], dtype=np.int64)
        col_enc = self._pad_sequences(col_encs)
        col_name_len = None
        col_len = np.array(col_lens, dtype=np.int64)

        return EncoderOutput(q_enc, q_len, col_enc, col_name_len, col_len)

    def forward(self, data, perm, st, ed, table_type):
        batch_size = ed - st

        descs = []
        for permuted_idx in perm[st:ed]:
            table_names, column_names, column_types, \
                column_to_table, table_to_column, foreign_keys, \
                foreign_keys_tables, primary_keys = data[permuted_idx]['ts']
            desc = {
                'question': data[permuted_idx]['question_tokens'],
                'columns': [[column_type] + column_name.split() for (table_id, column_name), column_type in zip(column_names, column_types)],
                'tables': [table_name.split() for table_name in table_names],
                'column_to_table': column_to_table,
                'table_to_column': table_to_column,
                'foreign_keys': foreign_keys,
                'foreign_keys_tables': foreign_keys_tables,
                'primary_keys': primary_keys,
            }
            descs.append(desc)

        q_enc, _ = self.spider_enc.question_encoder(
            self._lookup_embeddings_batched([[desc['question']] for desc in descs]))
        c_enc, c_boundaries = self.spider_enc.column_encoder(
            self._lookup_embeddings_batched([desc['columns'] for desc in descs]))
        t_enc, t_boundaries = self.spider_enc.table_encoder(
            self._lookup_embeddings_batched([desc['tables'] for desc in descs]))

        #q_enc_new, c_enc_new, t_enc_new = self.spider_enc.encs_update(descs, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)
        #q_enc, q_len = q_enc_new.pad(batch_first=True)
        #q_len = np.array(q_len, dtype=np.int64)
        #col_enc, col_len = c_enc_new.pad(batch_first=True)
        #col_name_len = None
        #col_len = np.array(col_len, dtype=np.int64)

        q_encs = []
        col_encs = []
        # Number of columns in each entry of batch
        col_lens = []
        for i, desc in enumerate(descs):
            q_enc_new, c_enc_new, t_enc_new = self.spider_enc.encs_update.forward_unbatched(
                desc,
                q_enc.select(i).unsqueeze(1),
                c_enc.select(i).unsqueeze(1),
                c_boundaries[i],
                t_enc.select(i).unsqueeze(1),
                t_boundaries[i])
            
            q_encs.append(q_enc_new)
            col_encs.append(c_enc_new)
            col_lens.append(len(desc['columns']))
        
        q_enc = self._pad_sequences(q_encs)
        q_len = np.array([q_enc_elem.shape[1] for q_enc_elem in q_encs], dtype=np.int64)
        col_enc = self._pad_sequences(col_encs)
        col_name_len = None
        col_len = np.array(col_lens, dtype=np.int64)
        return EncoderOutput(q_enc, q_len, col_enc, col_name_len, col_len)
