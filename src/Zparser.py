import functools

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
#import src_dep_const_test.chart_helper as chart_helper
from Decoder.uniform_decoder import Uniform_Decoder
import makehp
import utils

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"
CHAR_PAD = "\5"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    }


class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

#
class FeatureDropoutFunction(nn.functional._functions.dropout.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None, None
        else:
            return grad_output, None, None, None, None

#
class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

#
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

#
class ScaledAttention(nn.Module):
    def __init__(self, hparams, attention_dropout=0.1):
        super(ScaledAttention, self).__init__()
        self.hparams = hparams
        self.temper = hparams.d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper


        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

#
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, hparams, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hparams = hparams

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledAttention(hparams, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1)) # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
                ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.data.new(n_head, mb_size, len_padded, d_k).fill_(0.)
        k_padded = k_s.data.new(n_head, mb_size, len_padded, d_k).fill_(0.)
        v_padded = v_s.data.new(n_head, mb_size, len_padded, d_v).fill_(0.)
        q_padded = Variable(q_padded)
        k_padded = Variable(k_padded)
        v_padded = Variable(v_padded)
        invalid_mask = torch_t.ByteTensor(mb_size, len_padded).fill_(True)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
            )

    def combine_v(self, outputs):
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask
            )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded

#
class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()


    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            num_embeddings_list,
            d_embedding,
            hparams,
            d_positional=None,
            max_len=300,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            emb_dropouts_list=None,
            extra_content_dropout=None,
            word_table_np = None,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None
        self.hparams = hparams

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        if word_table_np is not None:
            self.pretrain_dim = word_table_np.shape[1]
        else:
            self.pretrain_dim = 0

        embs = []
        emb_dropouts = []
        cun = len(num_embeddings_list)*2 #must use chars emb or lm models
        for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            if hparams.use_cat:
                if i == len(num_embeddings_list) - 1:
                    #last is word
                    emb = nn.Embedding(num_embeddings, self.d_content//cun - self.pretrain_dim, **kwargs)
                else :
                    emb = nn.Embedding(num_embeddings, self.d_content//cun, **kwargs)
            else :
                emb = nn.Embedding(num_embeddings, self.d_content - self.pretrain_dim, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)

        if word_table_np is not None:
            self.pretrain_emb = nn.Embedding(word_table_np.shape[0], self.pretrain_dim)
            self.pretrain_emb.weight.data.copy_(torch.from_numpy(word_table_np))
            self.pretrain_emb.weight.requires_grad_(False)
            self.pretrain_emb_dropout = FeatureDropout(0.33)

        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)
        if not self.hparams.use_lstm:
            # Learned embeddings
            self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, self.d_positional))
            init.normal_(self.position_table)

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
            ]
        if self.hparams.use_cat:
            content_annotations = torch.cat(content_annotations, dim = -1)
        else :
            content_annotations = sum(content_annotations)
        if self.pretrain_dim != 0:
            content_annotations = torch.cat([content_annotations, self.pretrain_emb_dropout(self.pretrain_emb(pre_words_idxs), batch_idxs)], dim  = 1)

        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                extra_content_annotations = self.extra_content_dropout(extra_content_annotations, batch_idxs)

            if self.hparams.use_cat:
                content_annotations = torch.cat(
                    [content_annotations, extra_content_annotations], dim=-1)
            else:
                content_annotations += extra_content_annotations

        # Combine the content and timing signals
        annotations = None
        if not self.hparams.use_lstm:
            timing_signal = torch.cat([self.position_table[:seq_len,:] for seq_len in batch_idxs.seq_lens_np], dim=0)
            timing_signal = self.timing_dropout(timing_signal, batch_idxs)
            if self.partitioned:
                annotations = torch.cat([content_annotations, timing_signal], 1)
            else:
                annotations = content_annotations + timing_signal

            #print(annotations.shape)
            annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

        else:
            annotations = None

        content_annotations = self.dropout(content_annotations, batch_idxs)

        return annotations, content_annotations, batch_idxs

#

class CharacterLSTM(nn.Module):
    def __init__(self, num_embeddings, d_embedding, d_out,
            char_dropout=0.0,
            normalize=False,
            **kwargs):
        super(CharacterLSTM, self).__init__()

        self.d_embedding = d_embedding
        self.d_out = d_out

        self.lstm = nn.LSTM(self.d_embedding, self.d_out // 2, num_layers=1, bidirectional=True)

        self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
        self.char_dropout = nn.Dropout(char_dropout)

        if normalize:
            print("This experiment: layer-normalizing after character LSTM")
            self.layer_norm = LayerNormalization(self.d_out, affine=False)
        else:
            self.layer_norm = lambda x: x

    def forward(self, chars_padded_np, word_lens_np, batch_idxs):
        # copy to ensure nonnegative stride for successful transfer to pytorch
        decreasing_idxs_np = np.argsort(word_lens_np)[::-1].copy()
        decreasing_idxs_torch = from_numpy(decreasing_idxs_np)
        decreasing_idxs_torch.requires_grad_(False)

        chars_padded = from_numpy(chars_padded_np[decreasing_idxs_np])
        chars_padded.requires_grad_(False)
        word_lens = from_numpy(word_lens_np[decreasing_idxs_np])

        inp_sorted = nn.utils.rnn.pack_padded_sequence(chars_padded, word_lens_np[decreasing_idxs_np], batch_first=True)
        inp_sorted_emb = nn.utils.rnn.PackedSequence(
            self.char_dropout(self.emb(inp_sorted.data)),
            inp_sorted.batch_sizes)
        _, (lstm_out, _) = self.lstm(inp_sorted_emb)

        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        # Undo sorting by decreasing word length
        res = torch.zeros_like(lstm_out)
        res.index_copy_(0, decreasing_idxs_torch, lstm_out)

        res = self.layer_norm(res)
        return res

def get_elmo_class():
    # Avoid a hard dependency by only importing Elmo if it's being used
    from allennlp.modules.elmo import Elmo

    class ModElmo(Elmo):
       def forward(self, inputs):
            """
            Unlike Elmo.forward, return vector representations for bos/eos tokens

            This modified version does not support extra tensor dimensions

            Parameters
            ----------
            inputs : ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

            Returns
            -------
            Dict with keys:
            ``'elmo_representations'``: ``List[torch.autograd.Variable]``
                A ``num_output_representations`` list of ELMo representations for the input sequence.
                Each representation is shape ``(batch_size, timesteps + 2, embedding_dim)``
            ``'mask'``:  ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.
            """
            # reshape the input if needed
            original_shape = inputs.size()
            timesteps, num_characters = original_shape[-2:]
            assert len(original_shape) == 3, "Only 3D tensors supported here"
            reshaped_inputs = inputs

            # run the biLM
            bilm_output = self._elmo_lstm(reshaped_inputs)
            layer_activations = bilm_output['activations']
            mask_with_bos_eos = bilm_output['mask']

            # compute the elmo representations
            representations = []
            for i in range(len(self._scalar_mixes)):
                scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
                # We don't remove bos/eos here!
                representations.append(self._dropout(representation_with_bos_eos))

            mask = mask_with_bos_eos
            elmo_representations = representations

            return {'elmo_representations': elmo_representations, 'mask': mask}
    return ModElmo


def get_bert(bert_model, bert_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pretrained_bert import BertTokenizer, BertModel
    if bert_model.endswith('.tar.gz'):
        tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    bert = BertModel.from_pretrained(bert_model)
    return tokenizer, bert

class Encoder(nn.Module):
    def __init__(self, hparams, embedding,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024,
                    d_positional=None,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding
        self.d_model = d_model
        self.hparams = hparams

        d_k = d_v = d_kv

        if hparams.use_lstm:
            self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True,
                                num_layers=hparams.num_layers, dropout=0.33)
        else:
            self.stacks = []

            for i in range(hparams.num_layers):
                attn = MultiHeadAttention(hparams, num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                          attention_dropout=attention_dropout, d_positional=d_positional)
                if d_positional is None:
                    ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                                 residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                            residual_dropout=residual_dropout)

                self.add_module(f"attn_{i}", attn)
                self.add_module(f"ff_{i}", ff)

                self.stacks.append((attn, ff))

    def word_pad(self,emb,batch_idxs):

        word_lens_np = batch_idxs.seq_lens_np
        max_len = batch_idxs.max_len
        batch_size = batch_idxs.batch_size

        words_padded = emb.data.new(max_len, batch_size, self.hparams.d_model).fill_(0.)

        batch_start = batch_idxs.boundaries_np[:-1]
        batch_end = batch_idxs.boundaries_np[1:]

        for i ,(start,end) in enumerate(zip(batch_start, batch_end)):
            words_padded[:end - start, i,  :] = emb[start:end , :]

        return words_padded     #(len, batch, d_model)


    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        emb = self.embedding_container[0]
        res, res_c, batch_idxs = emb(xs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)
        if self.hparams.use_lstm:
            words_padded = self.word_pad(res_c, batch_idxs) # (len, batch, d_model)
            #res = words_padded
            lstm_out, (h_n, h_c) = self.lstm(torch.transpose(words_padded, 0, 1)) # (batch, len ,d_model)

            assert lstm_out.shape == (batch_idxs.batch_size, batch_idxs.max_len, self.d_model)
            #lstm_out = res_out
            batch_start = batch_idxs.boundaries_np[:-1]
            batch_end = batch_idxs.boundaries_np[1:]
            out_list = []

            for i, (start, end) in enumerate(zip(batch_start, batch_end)):
                out_list.append(lstm_out[i, :end - start, :])

            if len(out_list) > 1:
                res = torch.cat(out_list, dim=0)
            else:
                res = lstm_out.view(-1, lstm_out.size(-1))

        else:
            for i, (attn, ff) in enumerate(self.stacks):
                res, current_attns = attn(res, batch_idxs)
                res = ff(res, batch_idxs)

        return res, batch_idxs


class ChartParser(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            srl_vocab,
            hparams,
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab
        self.srl_vocab = srl_vocab

        self.hparams = hparams
        self.d_model = hparams.d_model

        if hparams.use_lstm:
            self.d_content = self.d_model
            self.d_positional = None
            self.partitioned = False
        else:
            self.partitioned = hparams.partitioned
            self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
            self.d_positional = (hparams.d_model // 2) if self.partitioned else None

        num_embeddings_map = {
            'tags': tag_vocab.size,
            'words': word_vocab.size,
            'chars': char_vocab.size,
        }
        emb_dropouts_map = {
            'tags': hparams.tag_emb_dropout,
            'words': hparams.word_emb_dropout,
        }

        self.emb_types = []
        if hparams.use_tags:
            self.emb_types.append('tags')
        if hparams.use_words:
            self.emb_types.append('words')

        self.use_tags = hparams.use_tags

        self.morpho_emb_dropout = None

        self.char_encoder = None
        self.elmo = None
        self.bert = None
        ex_dim = self.d_content
        if self.hparams.use_cat:
            cun = 0
            if hparams.use_words or hparams.use_tags:
                ex_dim = ex_dim // 2 #word dim = self.d_content/2
            if hparams.use_chars_lstm:
                cun = cun+1
            if hparams.use_elmo or hparams.use_bert:
                cun = cun + 1
            if cun > 0 :
                ex_dim = ex_dim // cun
        if hparams.use_chars_lstm:
            self.char_encoder = CharacterLSTM(
                num_embeddings_map['chars'],
                hparams.d_char_emb,
                ex_dim,
                char_dropout=hparams.char_lstm_input_dropout,
            )
        if hparams.use_elmo:
            self.elmo = get_elmo_class()(
                options_file="data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                weight_file="data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                num_output_representations=1,
                requires_grad=False,
                do_layer_norm=False,
                dropout=hparams.elmo_dropout,
                )
            d_elmo_annotations = 1024

            # Don't train gamma parameter for ELMo - the projection can do any
            # necessary scaling
            self.elmo.scalar_mix_0.gamma.requires_grad = False

            # Reshapes the embeddings to match the model dimension, and making
            # the projection trainable appears to improve parsing accuracy
            self.project_elmo = nn.Linear(d_elmo_annotations, ex_dim, bias=False)

        if hparams.use_bert :
            self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case)
            if hparams.bert_transliterate:
                from transliterate import TRANSLITERATIONS
                self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
            else:
                self.bert_transliterate = None

            d_bert_annotations = self.bert.pooler.dense.in_features
            self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

            self.project_bert = nn.Linear(d_bert_annotations, ex_dim, bias=False)

        if hparams.embedding_type != 'random' and hparams.use_words:
            embedd_dict, embedd_dim = utils.load_embedding_dict(hparams.embedding_type, hparams.embedding_path)
            scale = np.sqrt(3.0 / embedd_dim)
            table = np.zeros([word_vocab.size, embedd_dim], dtype=np.float32)
            oov = 0
            for index, word in enumerate(word_vocab.indices):
                if word in embedd_dict:
                    embedding = embedd_dict[word]
                else:
                    embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                    oov += 1
                table[index, :embedd_dim] = embedding
            print('oov: %d' % oov)
            word_table_np = table
            # self.project_pretrain = nn.Linear(embedd_dim, self.d_content, bias=False)
        else:
            word_table_np = None

        self.embedding = MultiLevelEmbedding(
            [num_embeddings_map[emb_type] for emb_type in self.emb_types],
            hparams.d_model,
            hparams=hparams,
            d_positional=self.d_positional,
            dropout=hparams.embedding_dropout,
            timing_dropout=hparams.timing_dropout,
            emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
            extra_content_dropout=self.morpho_emb_dropout,
            max_len=hparams.sentence_max_len,
            word_table_np=word_table_np,
        )

        self.encoder = Encoder(
            hparams,
            self.embedding,
            num_layers=hparams.num_layers,
            num_heads=hparams.num_heads,
            d_kv=hparams.d_kv,
            d_ff=hparams.d_ff,
            d_positional=self.d_positional,
            relu_dropout=hparams.relu_dropout,
            residual_dropout=hparams.residual_dropout,
            attention_dropout=hparams.attention_dropout,
        )

        self.decoder = Uniform_Decoder(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            srl_vocab,
            hparams
        )

        if use_cuda:
            self.cuda()

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'd_prespan' not in hparams:
            hparams['d_prespan'] = 512
        if 'use_prespan_ff' not in hparams:
            hparams['use_prespan_ff'] = False
        if 'joint_syn' not in hparams:
            hparams['joint_syn'] = True
        if 'joint_srl' not in hparams:
            hparams['joint_srl'] = True
        if 'joint_srl_dep' not in hparams:
            hparams['joint_srl_dep'] = hparams['joint_srl']
        if 'joint_srl_span' not in hparams:
            hparams['joint_srl_span'] = hparams['joint_srl']
        if 'joint_syn_const' not in hparams:
            hparams['joint_syn_const'] = hparams['joint_syn']
        if 'joint_syn_dep' not in hparams:
            hparams['joint_syn_dep'] = hparams['joint_syn']
        if 'use_gold_predicate' not in hparams:
            hparams['use_gold_predicate'] = False
        if 'use_srl_biaffine' not in hparams:
            hparams['use_srl_biaffine'] = False
        if 'use_gold_predicate' not in hparams:
            hparams['use_gold_predicate'] = False
        if 'use_srl_jointdecode' not in hparams:
            hparams['use_srl_jointdecode'] = False
        if 'joint_pos' not in hparams:
            hparams['joint_pos'] = False


        spec['hparams'] = makehp.HParams(**hparams)
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        if not hparams['use_elmo']:
            res.load_state_dict(model)
        else:
            state = {k: v for k,v in res.state_dict().items() if k not in model}
            state.update(model)
            res.load_state_dict(state)
        if use_cuda:
            res.cuda()
        return res

    def split_batch(self, sentences, golds, srlspans, srldeps, subbatch_max_tokens=3000):
        lens = [len(sentence) + 2 for sentence in sentences]

        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]], \
                      [srlspans[i] for i in lens_argsort[:subbatch_size]], [srldeps[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse_batch(self, sentences, gold_trees=None, gold_srlspans = None, gold_srldeps = None, gold_verbs = None, syndep_heads = None):
        is_train = gold_trees is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)

        if gold_trees is None:
            gold_trees = [None] * len(sentences)
            gold_srlspans = [None] * len(sentences)
            gold_srldeps = [None] * len(sentences)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        word_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            #print(sentence)
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_idxs[i] = 0 if not self.use_tags else self.tag_vocab.index_or_unk(tag, TAG_UNK)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_idxs[i] = self.word_vocab.index(word)
                batch_idxs[i] = snum
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs,
            'words': word_idxs,
        }
        emb_idxs = [
            from_numpy(emb_idxs_map[emb_type]).requires_grad_(False)
            for emb_type in self.emb_types
        ]
        pre_words_idxs = from_numpy(word_idxs).requires_grad_(False)

        extra_content_annotations_list = []
        extra_content_annotations = None
        if self.char_encoder is not None:
            assert isinstance(self.char_encoder, CharacterLSTM)
            max_word_len = max([max([len(word) for tag, word in sentence]) for sentence in sentences])
            # Add 2 for start/stop tokens
            max_word_len = max(max_word_len, 3) + 2
            char_idxs_encoder = np.zeros((packed_len, max_word_len), dtype=int)
            word_lens_encoder = np.zeros(packed_len, dtype=int)
            i = 0
            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate([(START, START)] + sentence + [(STOP, STOP)]):
                    j = 0
                    char_idxs_encoder[i, j] = self.char_vocab.index(CHAR_START_WORD)
                    j += 1
                    if word in (START, STOP):
                        char_idxs_encoder[i, j:j + 3] = self.char_vocab.index(
                            CHAR_START_SENTENCE if (word == START) else CHAR_STOP_SENTENCE
                        )
                        j += 3
                    else:
                        for char in word:
                            char_idxs_encoder[i, j] = self.char_vocab.index_or_unk(char, CHAR_UNK)
                            j += 1
                    char_idxs_encoder[i, j] = self.char_vocab.index(CHAR_STOP_WORD)
                    word_lens_encoder[i] = j + 1
                    i += 1
            assert i == packed_len

            extra_content_annotations_list.append(self.char_encoder(char_idxs_encoder, word_lens_encoder, batch_idxs))
        if self.elmo is not None:
            # See https://github.com/allenai/allennlp/blob/c3c3549887a6b1fb0bc8abf77bc820a3ab97f788/allennlp/data/token_indexers/elmo_indexer.py#L61
            # ELMO_START_SENTENCE = 256
            # ELMO_STOP_SENTENCE = 257
            ELMO_START_WORD = 258
            ELMO_STOP_WORD = 259
            ELMO_CHAR_PAD = 260

            # Sentence start/stop tokens are added inside the ELMo module
            max_sentence_len = max([(len(sentence)) for sentence in sentences])
            max_word_len = 50
            char_idxs_encoder = np.zeros((len(sentences), max_sentence_len, max_word_len), dtype=int)

            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate(sentence):
                    char_idxs_encoder[snum, wordnum, :] = ELMO_CHAR_PAD

                    j = 0
                    char_idxs_encoder[snum, wordnum, j] = ELMO_START_WORD
                    j += 1
                    assert word not in (START, STOP)
                    for char_id in word.encode('utf-8', 'ignore')[:(max_word_len-2)]:
                        char_idxs_encoder[snum, wordnum, j] = char_id
                        j += 1
                    char_idxs_encoder[snum, wordnum, j] = ELMO_STOP_WORD

                    # +1 for masking (everything that stays 0 is past the end of the sentence)
                    char_idxs_encoder[snum, wordnum, :] += 1

            char_idxs_encoder = from_numpy(char_idxs_encoder).requires_grad_(False)

            elmo_out = self.elmo.forward(char_idxs_encoder)
            elmo_rep0 = elmo_out['elmo_representations'][0]
            elmo_mask = elmo_out['mask']

            elmo_annotations_packed = elmo_rep0[elmo_mask.byte()].view(packed_len, -1)

            # Apply projection to match dimensionality
            extra_content_annotations_list.append(self.project_elmo(elmo_annotations_packed))

        if self.bert is not None:
            all_input_ids = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_input_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_word_start_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_word_end_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)

            subword_max_len = 0
            for snum, sentence in enumerate(sentences):
                tokens = []
                word_start_mask = []
                word_end_mask = []

                tokens.append("[CLS]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                if self.bert_transliterate is None:
                    cleaned_words = []
                    for _, word in sentence:
                        word = BERT_TOKEN_MAPPING.get(word, word)
                        if word == "n't" and cleaned_words:
                            cleaned_words[-1] = cleaned_words[-1] + "n"
                            word = "'t"
                        cleaned_words.append(word)
                else:
                    # When transliterating, assume that the token mapping is
                    # taken care of elsewhere
                    cleaned_words = [self.bert_transliterate(word) for _, word in sentence]

                for word in cleaned_words:
                    word_tokens = self.bert_tokenizer.tokenize(word)
                    if len(word_tokens) ==0:
                        word_tokens = ["[UNK]"]
                    for _ in range(len(word_tokens)):
                        word_start_mask.append(0)
                        word_end_mask.append(0)
                    word_start_mask[len(tokens)] = 1
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)
                tokens.append("[SEP]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(input_ids))

                all_input_ids[snum, :len(input_ids)] = input_ids
                all_input_mask[snum, :len(input_mask)] = input_mask
                all_word_start_mask[snum, :len(word_start_mask)] = word_start_mask
                all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

            all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
            all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
            all_word_start_mask = from_numpy(np.ascontiguousarray(all_word_start_mask[:, :subword_max_len]))
            all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
            all_encoder_layers, _ = self.bert(all_input_ids, attention_mask=all_input_mask)
            del _
            features = all_encoder_layers[-1]

            features_packed = features.masked_select(all_word_end_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1,
                                                                                                              features.shape[
                                                                                                                  -1])

            # For now, just project the features from the last word piece in each word
            extra_content_annotations = self.project_bert(features_packed)

        if len(extra_content_annotations_list) > 1 :
            if self.hparams.use_cat:
                extra_content_annotations = torch.cat(extra_content_annotations_list, dim = -1)
            else:
                extra_content_annotations = sum(extra_content_annotations_list)
        elif len(extra_content_annotations_list) == 1:
            extra_content_annotations = extra_content_annotations_list[0]

        annotations, _ = self.encoder(emb_idxs, pre_words_idxs, batch_idxs,
                                      extra_content_annotations=extra_content_annotations)

        if self.partitioned and not self.hparams.use_lstm:
            annotations = torch.cat([
                annotations[:, 0::2],
                annotations[:, 1::2],
            ], 1)

        if self.hparams.use_bispan_respresent:
            fencepost_annotations = torch.cat([
                annotations[:-1, :self.d_model // 2],
                - annotations[1:, self.d_model // 2:],
            ], 1)
        else:
            fencepost_annotations = torch.cat([
                annotations[:-1, :self.d_model // 2],
                annotations[1:, self.d_model // 2:],
            ], 1)

        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations


        if not is_train:

            decoder_args = dict(
                fencepost_annotations_start=fencepost_annotations_start,
                fencepost_annotations_end=fencepost_annotations_end,
                batch_idxs=batch_idxs,
                sentences=sentences,
                gold_verbs = gold_verbs
            )

            syntree_pred, score_list, srlspan_pred, srldep_pred = self.decoder.decode(annotations, **decoder_args)

            return syntree_pred, srlspan_pred, srldep_pred
        else:

            loss_args = dict(
                fencepost_annotations_start = fencepost_annotations_start,
                fencepost_annotations_end =fencepost_annotations_end,
                batch_idxs =batch_idxs,
                sentences =sentences,
                gold_trees = gold_trees,
                gold_srlspans = gold_srlspans,
                gold_srldeps = gold_srldeps)

            loss, srl_loss, synconst_loss, syndep_loss = self.decoder.cal_loss(annotations, **loss_args)

            return loss, srl_loss, synconst_loss, syndep_loss