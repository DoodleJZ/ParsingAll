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
import Decoder.hpsg_decoder as hpsg_decoder
import Decoder.synconst_scorer as synconst_scorer
import Decoder.srl_helper as srl_decoder
import Decoder.srlspan_helper as srlspan_helper
import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

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
class Synconst_score(nn.Module):
    def __init__(self, hparams, synconst_vocab):
        super(Synconst_score, self).__init__()

        self.hparams = hparams
        input_dim = hparams.d_model
        if self.hparams.use_syncatspan:
            input_dim = input_dim * 2

        self.f_label = nn.Sequential(
            nn.Linear(input_dim, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, synconst_vocab.size - 1),
        )

    def label_score(self, span_rep):
        return self.f_label(span_rep)

    def forward(self, fencepost_annotations_start, fencepost_annotations_end):
        if self.hparams.use_syncatspan:
            n = fencepost_annotations_end.size(0)

            fencepost_annotations_end = fencepost_annotations_end.repeat(n, 1)
            fencepost_annotations_end = fencepost_annotations_end.view(n, n, -1)

            fencepost_annotations_start = fencepost_annotations_start.repeat(1, n)
            fencepost_annotations_start = fencepost_annotations_start.view(n, n, -1)

            span_features = torch.cat([fencepost_annotations_start, fencepost_annotations_end], dim=-1)

        else:
            span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                             - torch.unsqueeze(fencepost_annotations_start, 1))

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
            ], 2)

        return label_scores_chart

class BiLinear(nn.Module):
    '''
    Bi-linear layer
    '''
    def __init__(self, left_features, right_features, out_features, bias=True):
        '''

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        '''

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        '''
        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(-1, self.left_features)
        input_right = input_right.view(-1, self.right_features)

        # output [batch, out_features]
        output = nn.functional.bilinear(input_left, input_right, self.U, self.bias)
        output = output + nn.functional.linear(input_left, self.W_l, None) + nn.functional.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output
#
class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, hparams):
        super(BiAAttention, self).__init__()
        self.hparams = hparams

        self.dep_weight = nn.Parameter(torch_t.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
        nn.init.xavier_uniform_(self.dep_weight)

    def forward(self, input_d, input_e, input_s = None):

        score = torch.matmul(torch.cat(
            [input_d, torch_t.FloatTensor(input_d.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), self.dep_weight)
        score1 = torch.matmul(score, torch.transpose(torch.cat(
            [input_e, torch_t.FloatTensor(input_e.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), 0, 1))

        return score1

class Dep_score(nn.Module):
    def __init__(self, hparams, num_labels):
        super(Dep_score, self).__init__()

        self.dropout_out = nn.Dropout2d(p=0.33)
        self.hparams = hparams
        out_dim = hparams.d_biaffine#d_biaffine
        self.arc_h = nn.Linear(hparams.d_model, hparams.d_biaffine)
        self.arc_c = nn.Linear(hparams.d_model, hparams.d_biaffine)

        self.attention = BiAAttention(hparams)

        self.type_h = nn.Linear(hparams.d_model, hparams.d_label_hidden)
        self.type_c = nn.Linear(hparams.d_model, hparams.d_label_hidden)
        self.bilinear = BiLinear(hparams.d_label_hidden, hparams.d_label_hidden, num_labels)

    def forward(self, outputs, outpute):
        # output from rnn [batch, length, hidden_size]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
        outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)

        # output size [batch, length, arc_space]
        arc_h = nn.functional.relu(self.arc_h(outputs))
        arc_c = nn.functional.relu(self.arc_c(outpute))

        # output size [batch, length, type_space]
        type_h = nn.functional.relu(self.type_h(outputs))
        type_c = nn.functional.relu(self.type_c(outpute))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=0)
        type = torch.cat([type_h, type_c], dim=0)

        arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
        arc_h, arc_c = arc.chunk(2, 0)

        type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
        type_h, type_c = type.chunk(2, 0)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        out_arc = self.attention(arc_h, arc_c)
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type


class Srl_score(nn.Module):
    def __init__(self, hparams, srl_vocab):
        super(Srl_score, self).__init__()

        self.hparams = hparams
        self.srl_vocab = srl_vocab

        self.dropout_out = nn.Dropout2d(p=0.3)
        d_verb = hparams.d_model
        if self.hparams.use_verb_ff:

            self.f_verb = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_verb),
                nn.ReLU(),
            )
            d_verb = hparams.d_verb

        self.f_verbscore = nn.Sequential(
            nn.Linear(d_verb, hparams.d_score_hidden),
            LayerNormalization(hparams.d_score_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_score_hidden, 1),
        )

        d_prespan = hparams.d_model
        if self.hparams.use_prespan_ff:
            self.f_span_start = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_prespan),
                nn.ReLU(),
            )
            self.f_span_end = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_prespan),
                nn.ReLU(),
            )
            d_prespan = hparams.d_prespan


        if self.hparams.use_catspan:
            d_prespan = d_prespan * 2

        d_span = d_prespan

        if self.hparams.use_span_ff:
            self.f_span = nn.Sequential(
                nn.Linear(d_prespan, hparams.d_span),
                nn.ReLU(),
            )
            d_span = hparams.d_span

        self.f_spanscore = nn.Sequential(
            nn.Linear(d_span, hparams.d_score_hidden),
            LayerNormalization(hparams.d_score_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_score_hidden, 1),
        )

        if hparams.use_srl_biaffine:
            self.f_bilinear = torch.nn.Bilinear(d_verb, d_span, self.srl_vocab.size - 1, bias=False)
            self.f_linear = torch.nn.Linear(d_verb + d_span, self.srl_vocab.size - 1,)
        elif self.hparams.use_srl_dot:
            assert d_span == d_verb
            self.f_srl_label = nn.Sequential(
                nn.Linear(d_span, hparams.d_label_hidden),
                LayerNormalization(hparams.d_label_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_label_hidden, self.srl_vocab.size - 1),
            )
        else:

            self.f_srl_label = nn.Sequential(
                nn.Linear(d_span + d_verb, hparams.d_label_hidden),
                LayerNormalization(hparams.d_label_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_label_hidden, self.srl_vocab.size - 1),
            )

    def span_list_mat(self, span_id, len):

        span_id_mat = -np.ones((len,len),dtype=int)
        span_id_list = []
        for i, id in enumerate(span_id):
            left = id // len
            right = id % len
            span_id_list.append((left,right))
            span_id_mat[left][right] = i    #right - left

        return span_id_list, span_id_mat

    def cat_span(self, p_start, p_end):

        n = p_start.size(0)

        p_end = p_end.repeat(n, 1)
        p_end = p_end.view(n, n, -1)

        p_start = p_start.repeat(1, n)
        p_start = p_start.view(n, n, -1)

        span_v = torch.cat([p_start, p_end], dim=-1)

        return span_v


    def forward(self, p_start, p_end, verb_v, gold_verb_list = None, remove_span = False, select_span_score = None, select_arc_score = None):
        '''

        :param start_v: [<start>+n, dim]
        :param end_v: [<start>+n, dim]
        :param verb_v: [<start>+n, dim]
        :param select_span_score: numpy, leng*leng, syconst score
        :param select_arc_score: numpy, leng*leng, syndep score
        :return:
        srl_label_chart: [ <start>+n, <start>+n, label]
        verb_id: start from 0, without <start>
        span_list:
        span_mat:
        '''

        leng = p_start.size(0)
        num_verb = max(int(self.hparams.labmda_verb * (leng - 1)), 1)
        num_verb = min(num_verb, self.hparams.max_num_verb)
        num_span = min(int(self.hparams.labmda_span * leng*leng), self.hparams.max_num_span)

        p_start = self.dropout_out(p_start.transpose(1, 0)).transpose(1, 0)
        p_end = self.dropout_out(p_end.transpose(1, 0)).transpose(1, 0)
        verb_v = self.dropout_out(verb_v.transpose(1, 0)).transpose(1, 0)

        # make verb
        verb_v = verb_v[1:, :]  # remove <start>
        if self.hparams.use_verb_ff:
            verb_v = self.f_verb(verb_v)
            verb_v = self.dropout_out(verb_v.transpose(1, 0)).transpose(1, 0)

        verb_score =None
        if self.hparams.use_gold_predicate:
            verb_id = from_numpy(np.array(gold_verb_list)).long()
            verb_v = verb_v[verb_id]
            verb_score = None
        else:
            verb_score = self.f_verbscore(verb_v)
            verb_score = torch.squeeze(verb_score, dim=-1)
            verb_sort = torch.topk(verb_score, num_verb)
            verb_topk_score = verb_sort[0]
            verb_id = verb_sort[1]
            verb_id, idces = torch.sort(verb_id)
            verb_v = verb_v[verb_id]
            verb_topk_score = verb_topk_score[idces]
            if num_verb > 1:
                verb_id = torch.squeeze(verb_id, dim = 0)

        verb_mask = -np.ones(leng, dtype= int)
        for i, verb_idx in enumerate(verb_id):
            verb_mask[verb_idx] = i

        if len(verb_id) == 0:  #no gold verb
            return None, verb_score, verb_id, verb_mask, None, None, None, None

        # make span
        if self.hparams.use_prespan_ff:
            p_start = self.f_span_start(p_start)
            p_end = self.f_span_end(p_end)
            p_start = self.dropout_out(p_start.transpose(1, 0)).transpose(1, 0)
            p_end = self.dropout_out(p_end.transpose(1, 0)).transpose(1, 0)

        if self.hparams.use_addspan:
            span_v = (torch.unsqueeze(p_end, 0)
                      + torch.unsqueeze(p_start, 1))
        elif self.hparams.use_catspan:

            span_v = self.cat_span(p_start, p_end)
        else:
            if not self.hparams.use_prespan_ff:
                d_span = p_end.size(1)
                p_end = torch.cat([p_end[:, d_span // 2:], p_end[:, : d_span // 2]], dim=-1)

            span_v = (torch.unsqueeze(p_end, 0)
                      - torch.unsqueeze(p_start, 1))
        #[(0,0)...(0,n),(1,0)....(1,n)....(n,0)...(n,n)]
        span_v = span_v.view(-1, span_v.size(2))
        if self.hparams.use_span_ff:
            span_v = self.f_span(span_v)
            span_v = self.dropout_out(span_v.transpose(1, 0)).transpose(1, 0)

        span_score = self.f_spanscore(span_v)
        span_score = torch.squeeze(span_score, dim=-1)

        span_sort = torch.topk(span_score, num_span)

        span_topk_score = span_sort[0]
        span_id = span_sort[1]
        span_id = torch.squeeze(span_id, dim = 0)
        if remove_span:
            #just remove single span for dep srl, since some train example do not have gold dep srl
            new_spanidx = []
            for i, span_idx in enumerate(span_id):
                left = span_idx // leng
                right = span_idx % leng
                if left != right:
                    new_spanidx.append(i)

            new_spanidx = from_numpy(np.array(new_spanidx))
            span_id = span_id[new_spanidx]
            span_topk_score = span_topk_score[new_spanidx]

        span_v = span_v[span_id]
        span_list, span_mat = self.span_list_mat(span_id.cpu().data.numpy(), leng)

        #make srl label
        #span_v:  [num_span, dim]
        #verb_v: [num_verb, dim]
        n = verb_v.size(0)
        m = span_v.size(0)

        verb_r = verb_v.repeat(m, 1)
        verb_r = verb_r.view(m, n, -1)

        span_r = span_v.repeat(1, n)
        span_r = span_r.view(m, n, -1)

        label_v = torch.cat([verb_r, span_r], dim=-1)
        #label_v: [num_span, num_verb, dim*2] = [m, n, dim*2]

        if self.hparams.use_srl_biaffine:
            srl_score = self.f_bilinear(verb_r, span_r)+ self.f_linear(label_v)
        elif self.hparams.use_srl_dot:
            srl_score = self.f_srl_label(torch.mul(verb_r, span_r))
        else:
            srl_score = self.f_srl_label(label_v)
        # srl_score: [num_span, num_verb, num_label]

        if not self.hparams.use_softmax_verb and not self.hparams.use_gold_predicate:
            srl_score = srl_score + verb_topk_score.view(1,-1,1).repeat(srl_score.size(0), 1, srl_score.size(2))
        if not self.hparams.use_softmax_span :
            srl_score = srl_score + span_topk_score.view(-1,1,1).repeat(1, srl_score.size(1), srl_score.size(2))

        srl_score = torch.cat([
            srl_score.new_zeros((srl_score.size(0), srl_score.size(1), 1)),
            srl_score
        ], 2)

        return srl_score, verb_score, verb_id, verb_mask, span_score, span_id, span_list, span_mat

class Uniform_Decoder(nn.Module):
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
        super(Uniform_Decoder, self).__init__()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab
        self.srl_vocab = srl_vocab

        self.hparams = hparams
        if self.hparams.joint_syn_const:
            self.synconst_f = Synconst_score(hparams, label_vocab)
        if self.hparams.joint_syn_dep:
            self.dep_score = Dep_score(hparams, type_vocab.size)
            self.loss_func = torch.nn.CrossEntropyLoss(size_average=False)
            self.loss_funt = torch.nn.CrossEntropyLoss(size_average=False)

        if self.hparams.joint_srl_span or self.hparams.joint_srl_dep:
            self.srl_f = Srl_score(hparams, srl_vocab)
            self.loss_srl = torch.nn.CrossEntropyLoss(size_average=False)

        if self.hparams.joint_pos:
            self.f_pos = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_label_hidden),
                LayerNormalization(hparams.d_label_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_label_hidden, tag_vocab.size),
            )
            self.loss_pos = torch.nn.CrossEntropyLoss(size_average=False)

    def cal_loss(self, annotations, fencepost_annotations_start, fencepost_annotations_end, batch_idxs, sentences,
                 gold_trees, gold_srlspans, gold_srldeps):

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        syndep_loss = 0
        synconst_loss = 0
        loss = 0
        if self.hparams.joint_syn_const:
            with torch.no_grad():
                for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                    label_scores_chart = self.synconst_f(fencepost_annotations_start[start:end,:],
                                                                            fencepost_annotations_end[start:end, :])
                    label_scores_chart_np = label_scores_chart.cpu().data.numpy()
                    decoder_args = dict(
                        sentence_len=len(sentences[i]),
                        label_scores_chart=label_scores_chart_np,
                        gold=gold_trees[i],
                        label_vocab=self.label_vocab,
                        is_train=True)
                    p_score, p_i, p_j, p_label, p_augment = synconst_scorer.decode(False, **decoder_args)
                    g_score, g_i, g_j, g_label, g_augment = synconst_scorer.decode(True, **decoder_args)
                    paugment_total += p_augment
                    num_p += p_i.shape[0]
                    pis.append(p_i + start)
                    pjs.append(p_j + start)
                    gis.append(g_i + start)
                    gjs.append(g_j + start)
                    plabels.append(p_label)
                    glabels.append(g_label)

            cells_i = from_numpy(np.concatenate(pis + gis))
            cells_j = from_numpy(np.concatenate(pjs + gjs))
            cells_label = from_numpy(np.concatenate(plabels + glabels))

            if self.hparams.use_syncatspan:
                cells_label_scores = self.synconst_f.label_score(torch.cat(
                    [fencepost_annotations_end[cells_j], fencepost_annotations_start[cells_i]],dim = -1))
            else:
                cells_label_scores = self.synconst_f.label_score(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
            cells_label_scores = torch.cat([
                cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
                cells_label_scores
            ], 1)
            cells_label_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
            loss = cells_label_scores[:num_p].sum() - cells_label_scores[num_p:].sum() + paugment_total
            synconst_loss = loss

        if self.hparams.joint_syn_dep:
            # syndep loss
            cun = 0
            for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                #[start,....,end-1]->[<root>,1, 2,...,n]
                leng = end - start
                arc_score, type_score = self.dep_score(annotations[start:end,:], annotations[start:end,:])
                # arc_score, type_score = self.dep_score(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:])
                arc_gather = [leaf.father for leaf in gold_trees[snum].leaves()]
                type_gather = [self.type_vocab.index(leaf.type) for leaf in gold_trees[snum].leaves()]
                cun += 1
                assert len(arc_gather) == leng - 1
                arc_score = torch.transpose(arc_score,0, 1)
                dep_loss = self.loss_func(arc_score[1:, :], from_numpy(np.array(arc_gather)).requires_grad_(False)) \
                       +  self.loss_funt(type_score[1:, :],from_numpy(np.array(type_gather)).requires_grad_(False))
                loss = loss +  dep_loss
                syndep_loss = syndep_loss + dep_loss

        srl_loss = 0
        if self.hparams.joint_srl_span or self.hparams.joint_srl_dep:
            # srl loss
            for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                srl_start = fencepost_annotations_start[start:end, :]
                srl_end = fencepost_annotations_end[start:end, :]
                srl_word = annotations[start:end, :]
                leng = end - start
                #[num_span, num_verb]
                srlspan_dict = gold_srlspans[snum]
                srldep_dict = gold_srldeps[snum]

                gold_verb_mask = np.zeros(leng - 1)

                if self.hparams.joint_srl_span:
                    for pred_id, gold_args in srlspan_dict.items():
                        #span srl verb start from 0
                        gold_verb_mask[pred_id] = 1
                if self.hparams.joint_srl_dep:
                    if srldep_dict is not None:
                        for pred_id, gold_args in srldep_dict.items():
                            gold_verb_mask[pred_id] = 1
                gold_verb_list = []
                for i in range(leng - 1):
                    if gold_verb_mask[i] == 1:
                        gold_verb_list.append(i)

                #srl_label_chart:[num_span, num_verb, num_label]

                srl_score_chart, verb_score, verb_id, verb_mask, span_score, span_id, span_list, span_mat = \
                    self.srl_f(srl_start, srl_end, srl_word, gold_verb_list = gold_verb_list, remove_span = srldep_dict is None)

                if len(verb_id) == 0:
                    #no verb, no loss
                    continue

                gold_span_np = np.zeros(leng * leng)
                g_srlspan = np.zeros((len(span_id), len(verb_id)))
                if self.hparams.joint_srl_span:
                    for pred_id, gold_args in srlspan_dict.items():
                        for a0 in gold_args:
                            left = a0[0]
                            right = a0[1] + 1       #span repr
                            gold_span_np[left * leng + right] = 1
                            span_idx = span_mat[left][right]
                            if span_idx != -1 and verb_mask[pred_id] != -1:
                                g_srlspan[span_idx, verb_mask[pred_id]] = self.srl_vocab.index(a0[2])

                if self.hparams.joint_srl_dep:
                    if srldep_dict is not None:
                        for pred_id, gold_args in srldep_dict.items():
                            for a0 in gold_args:
                                left = a0[0] + 1    #word from 1, 0 is <start>
                                right = left
                                span_idx = span_mat[left][right]
                                gold_span_np[left * leng + right] = 1
                                if span_idx != -1 and verb_mask[pred_id] != -1:
                                    g_srlspan[span_idx, verb_mask[pred_id]] = self.srl_vocab.index(a0[1])
                    else:
                        for left in range(1, leng):
                            gold_span_np[left * leng + left] = 1

                verb_loss = 0
                if self.hparams.use_softmax_verb:
                    assert not self.hparams.use_gold_predicate
                    gold_verb_mask = from_numpy(gold_verb_mask).byte()
                    verb_log_softmax = torch.nn.functional.log_softmax(verb_score, dim=-1)
                    verb_loss = - verb_log_softmax[gold_verb_mask].sum()

                span_loss = 0
                if self.hparams.use_softmax_span:
                    span_mask = from_numpy(gold_span_np).byte()
                    span_log_softmax = torch.nn.functional.log_softmax(span_score, dim=-1)
                    span_loss = - span_log_softmax[span_mask].sum()

                g_srlspan_ts = from_numpy(g_srlspan).long()

                srl_label_loss  = self.loss_srl(srl_score_chart.view(-1, srl_score_chart.size(2)), g_srlspan_ts.view(-1).requires_grad_(False))

                srl_loss += span_loss + verb_loss + srl_label_loss
                loss = loss + span_loss + verb_loss + srl_label_loss

        if self.hparams.joint_pos:
            for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                #[start,....,end-1]->[<root>,1, 2,...,n]
                leng = end - start
                pos_score = self.f_pos(annotations[start:end,:])
                pos_gather = [self.tag_vocab.index(leaf.goldtag) for leaf in gold_trees[snum].leaves()]
                assert len(pos_gather) == leng - 1
                pos_loss = self.loss_pos(pos_score[1:, :], from_numpy(np.array(pos_gather)).requires_grad_(False))
                loss = loss + pos_loss

        return loss, srl_loss, synconst_loss, syndep_loss

    def decode(self, annotations, fencepost_annotations_start, fencepost_annotations_end, batch_idxs, sentences, gold_verbs = None):

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if gold_verbs is None:
            gold_verbs = [None] * len(sentences)

        syntree_pred = []
        score_list = []
        srlspan_pred = []
        srldep_pred = []
        for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
            pos_pred = None
            if self.hparams.joint_pos:
                pos_score = self.f_pos(annotations[start:end, :])
                pos_score_np = pos_score.cpu().data.numpy()
                pos_score_np = pos_score_np[1:, :]  # remove root
                pos_pred = pos_score_np.argmax(axis=1)
                pos_pred = [self.tag_vocab.value(pos_pred_index) for pos_pred_index in pos_pred]

            if self.hparams.joint_syn_const or self.hparams.joint_syn_dep:
                syn_tree, score = self.hpsg_decoder(fencepost_annotations_start[start:end, :],
                                                      fencepost_annotations_end[start:end, :], annotations[start:end, :], sentences[i], pos_pred)

                syntree_pred.append(syn_tree)
                score_list.append(score)

            if self.hparams.joint_srl_span or self.hparams.joint_srl_dep:
                if self.hparams.use_srl_jointdecode:
                    srlspan_dict, srldep_dict = self.srl_decoder(fencepost_annotations_start[start:end, :],
                                                                 fencepost_annotations_end[start:end, :],
                                                                 annotations[start:end, :],
                                                                 gold_verbs[i])
                else:
                    srlspan_dict = None
                    srldep_dict = None
                    if self.hparams.joint_srl_span:
                        srlspan_dict = self.srlspan_decoder(fencepost_annotations_start[start:end, :],
                                                       fencepost_annotations_end[start:end, :], annotations[start:end, :], gold_verbs[i])
                    if self.hparams.joint_srl_dep:
                        srldep_dict = self.srldep_decoder(fencepost_annotations_start[start:end, :],
                                                   fencepost_annotations_end[start:end, :], annotations[start:end, :], gold_verbs[i])

                srlspan_pred.append(srlspan_dict)
                srldep_pred.append(srldep_dict)

        return syntree_pred, score_list, srlspan_pred, srldep_pred


    def srl_decoder(self, fencepost_annotations_start, fencepost_annotations_end, annotation, gold_verb = None):
        # change gold_verb to mask id
        leng = annotation.size(0)
        gold_verb_list = []
        if gold_verb is not None:
            for verb_span in gold_verb:
                gold_verb_list.append(verb_span[0])

        srl_score, verb_score, verb_id, verb_mask, span_score, span_id, span_list, span_mat = \
            self.srl_f(fencepost_annotations_start, fencepost_annotations_end, annotation,
                       gold_verb_list=gold_verb_list)

        srlspan_dict = {}
        srldep_dict = {}

        if len(verb_id) == 0:
            return srlspan_dict, srldep_dict

        verb_id_np = verb_id.cpu().data.numpy().astype(np.int32)
        span_mat = span_mat.astype(np.int32)
        srl_score_np = srl_score.cpu().data.numpy()

        # srl_dpdict = srlspan_decoder.decode(leng, verb_id_np, span_mat, srl_score_np, self.srl_vocab)

        for j, verb_idx in enumerate(verb_id_np):
            if self.hparams.use_gold_predicate:
                start = gold_verb[j][0]
                end = gold_verb[j][1]
                srlspan_dict[verb_idx] = []
                srldep_dict[verb_idx] = []
                srl_decoder.decode(0, start + 1, leng, verb_idx, span_mat, srl_score_np[:, j, :], srlspan_dict, srldep_dict,
                                      self.srl_vocab)
                srlspan_dict[verb_idx].append((start, end, "V"))
                #srldep_dict[verb_idx].append((start, "V"))
                srl_decoder.decode(end + 1, leng, leng, verb_idx, span_mat, srl_score_np[:, j, :], srlspan_dict, srldep_dict,
                                      self.srl_vocab)
            else:
                srl_decoder.decode(0, leng, leng, verb_idx, span_mat, srl_score_np[:, j, :], srlspan_dict, srldep_dict,
                                      self.srl_vocab)

        return srlspan_dict, srldep_dict

    def srlspan_decoder(self, fencepost_annotations_start, fencepost_annotations_end, annotation, gold_verb = None):

        #change gold_verb to mask id
        leng = annotation.size(0)
        gold_verb_list = []
        if gold_verb is not None:
            for verb_span in gold_verb:
                gold_verb_list.append(verb_span[0])

        srl_score, verb_score, verb_id, verb_mask, span_score, span_id, span_list, span_mat = \
            self.srl_f(fencepost_annotations_start, fencepost_annotations_end, annotation, gold_verb_list = gold_verb_list)

        srl_dpdict = {}

        if len(verb_id) == 0:
            return srl_dpdict

        verb_id_np = verb_id.cpu().data.numpy().astype(np.int32)
        verb_mask = verb_mask.astype(np.int32)
        span_mat = span_mat.astype(np.int32)
        srl_score_np = srl_score.cpu().data.numpy()

        #srl_dpdict = srlspan_decoder.decode(leng, verb_id_np, span_mat, srl_score_np, self.srl_vocab)

        for j, verb_idx in enumerate(verb_id_np):
            if self.hparams.use_gold_predicate:
                start = gold_verb[j][0]
                end = gold_verb[j][1]
                srl_dpdict[verb_idx] = []
                srlspan_helper.decode(0, start + 1, leng, verb_idx, span_mat, srl_score_np[:,j,:], srl_dpdict, self.srl_vocab)
                srl_dpdict[verb_idx].append((start, end, "V"))
                srlspan_helper.decode(end + 1, leng, leng, verb_idx, span_mat, srl_score_np[:,j,:], srl_dpdict, self.srl_vocab)
            else:
                srlspan_helper.decode(0, leng, leng, verb_idx, span_mat, srl_score_np[:, j, :], srl_dpdict,
                                      self.srl_vocab)

        return srl_dpdict

    def srldep_decoder(self, fencepost_annotations_start, fencepost_annotations_end, annotation, gold_verb = None):

        gold_verb_list = []
        if gold_verb is not None:
            for verb_span in gold_verb:
                gold_verb_list.append(verb_span[0])

        srldep_score, verb_score, dep_verb_id, dep_verb_mask, span_score, span_id, dep_span_list, dep_span_mat = \
                                    self.srl_f(fencepost_annotations_start, fencepost_annotations_end, annotation, gold_verb_list = gold_verb_list)
        srldep_dict = {}
        if gold_verb is not None:
            assert len(dep_verb_id) == len(gold_verb)
        if len(dep_verb_id) == 0:
            return srldep_dict

        dep_verb_id = dep_verb_id.cpu().data.numpy().astype(np.int32)
        srldep_score_np = srldep_score.cpu().data.numpy()

        leng = annotation.size(0)

        for j, verb_idx in enumerate(dep_verb_id):
            if self.hparams.use_gold_predicate:
                srldep_dict[verb_idx] = []
        for i, span_idx in enumerate(dep_span_list):
            left = span_idx[0] - 1      #word from 1, 0 is <start>
            right = span_idx[1] - 1
            if left != right or left < 0:
                continue
            for j, verb_idx in enumerate(dep_verb_id):

                max_label_idx = np.argmax(srldep_score_np[i, j])
                dep_label = self.srl_vocab.value(max_label_idx)
                if max_label_idx > 0 and dep_label[-1]!="*" and dep_label[0] != "V":
                    if verb_idx not in srldep_dict:
                        srldep_dict[verb_idx] = []
                    srldep_dict[verb_idx].append((left, self.srl_vocab.value(max_label_idx)))

        if self.hparams.use_gold_predicate:

            assert len(srldep_dict) == len(dep_verb_id)

        return srldep_dict

    def hpsg_decoder(self, fencepost_annotations_start, fencepost_annotations_end, annotation, sentence, pred_pos = None):

        leng = annotation.size(0)

        if self.hparams.joint_syn_const:
            label_scores_chart = self.synconst_f(fencepost_annotations_start, fencepost_annotations_end)
            label_scores_chart_np = label_scores_chart.cpu().data.numpy()
        else:
            label_scores_chart_np = np.zeros((leng, leng, self.label_vocab.size), dtype= np.float32)

        if self.hparams.joint_syn_dep:
            arc_score, type_score = self.dep_score(annotation, annotation)
            arc_score_dc = torch.transpose(arc_score, 0, 1)
            arc_dc_np = arc_score_dc.cpu().data.numpy()

            type_np = type_score.cpu().data.numpy()
            type_np = type_np[1:, :]  # remove root
            type = type_np.argmax(axis=1)

        else:
            arc_dc_np = np.zeros((leng, leng), dtype= np.float32)
            type = np.zeros(leng, dtype= int)

        score, p_i, p_j, p_label, p_father = hpsg_decoder.decode(sentence_len=len(sentence),
            label_scores_chart=label_scores_chart_np * self.hparams.const_lada,
            type_scores_chart=arc_dc_np * (1.0 - self.hparams.const_lada))
        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.

        #make arrange table, sort the verb id
        idx = -1

        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab.value(label_idx)
            if (i + 1) >= j:
                tag, word = sentence[i]
                pred_tag = None
                if pred_pos is not None:
                    pred_tag = pred_pos[i]
                tree = trees.LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(type[i]), pred_tag)
                if label:
                    assert label[0] != Sub_Head
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label and label[0] != Sub_Head:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree_list = make_tree()
        assert len(tree_list) == 1
        tree = tree_list[0]
        return tree.convert(), score