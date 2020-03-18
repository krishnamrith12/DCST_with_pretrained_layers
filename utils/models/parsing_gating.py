import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from ..nn import Embedding
from ..nn import BiAAttention, BiLinear
from utils.tasks import parse
from ..nn import utils
from utils.morph_tagger import crf, highway
import pdb
class BiAffine_Parser_Gated(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, use_pos, use_char, pos_dim, num_pos, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, num_arcs, arc_space, arc_tag_space, num_gates,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 biaffine=True, arc_decode='mst', initializer=None):
        super(BiAffine_Parser_Gated, self).__init__()
        self.num_gates = num_gates
        self.rnn_encoder = BiRecurrentConv_Encoder(word_dim, num_words, char_dim, num_chars, use_pos, use_char,
                                                   pos_dim, num_pos, num_filters,
                                                   kernel_size, rnn_mode, hidden_size,
                                                   num_layers, embedd_word=embedd_word,
                                                   embedd_char=embedd_char, embedd_pos=embedd_pos,
                                                   p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)
        if self.num_gates >= 2:
            self.extra_rnn_encoders = nn.ModuleDict([[str(i), BiRecurrentConv_Encoder(word_dim, num_words, char_dim, num_chars, use_pos, use_char,
                                                   pos_dim, num_pos, num_filters,
                                                   kernel_size, rnn_mode, hidden_size,
                                                   num_layers, embedd_word=embedd_word,
                                                   embedd_char=embedd_char, embedd_pos=embedd_pos,
                                                   p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)] for i in range(num_gates - 1)])
            self.gate = Gating(num_gates, 2 * hidden_size)
        else:
            self.extra_rnn_encoders = None
            self.gate = None
        self.parser = BiAffine_Parser_Decoder(hidden_size, num_arcs, arc_space, arc_tag_space, biaffine, p_out, arc_decode)

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        encoder_output, hn, mask, length = self.rnn_encoder(input_word, input_char, input_pos, mask, length, hx)
        if self.num_gates >= 2:
            len_extra_encoders = len(self.extra_rnn_encoders.keys())
            extra_enconder_outputs = [self.extra_rnn_encoders[str(i)](input_word, input_char, input_pos, mask, length, hx)[0] for i in range(len_extra_encoders)]
            rnns_output = self.gate(tuple([encoder_output] + extra_enconder_outputs))
        else:
            rnns_output = encoder_output
        out_arc, out_arc_tag = self.parser(rnns_output, mask)
        return out_arc, out_arc_tag, mask, length

    def loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None):
        # out_arc shape [batch_size, length, length]
        # out_arc_tag shape [batch_size, length, arc_tag_space]
        loss_arc, loss_arc_tag = self.parser.loss(out_arc, out_arc_tag, heads, arc_tags, mask, length)
        return loss_arc, loss_arc_tag

    def decode(self, out_arc, out_arc_tag, mask=None, length=None, leading_symbolic=0):
        heads_pred, arc_tags_pred, scores = self.parser.decode(out_arc, out_arc_tag, mask, length, leading_symbolic)
        return heads_pred, arc_tags_pred, scores

    def pre_loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None, use_log=True, temperature=1.0):
        out_arc, out_arc_tag = self.parser.pre_loss(out_arc, out_arc_tag, heads, arc_tags, mask, length, use_log, temperature)
        return out_arc, out_arc_tag

class BiAffine_Parser_Decoder(nn.Module):
    def __init__(self, hidden_size, num_arcs, arc_space, arc_tag_space, biaffine, p_out, arc_decode):
        super(BiAffine_Parser_Decoder, self).__init__()
        self.num_arcs = num_arcs
        self.arc_space = arc_space
        self.arc_tag_space = arc_tag_space
        self.out_dim = hidden_size * 2
        self.biaffine = biaffine
        self.p_out = p_out
        self.arc_decode = arc_decode
        self.dropout_out = nn.Dropout(self.p_out)
        self.arc_h = nn.Linear(self.out_dim, self.arc_space)
        self.arc_c = nn.Linear(self.out_dim, self.arc_space)
        self.attention = BiAAttention(self.arc_space, self.arc_space, 1, biaffine=biaffine)
        self.arc_tag_h = nn.Linear(self.out_dim, arc_tag_space)
        self.arc_tag_c = nn.Linear(self.out_dim, arc_tag_space)
        self.bilinear = BiLinear(arc_tag_space, arc_tag_space, num_arcs)

    def forward(self, input, mask):
        # apply dropout for output
        # [batch_size, length, hidden_size] --> [batch_size, hidden_size, length] --> [batch_size, length, hidden_size]
        input = self.dropout_out(input.transpose(1, 2)).transpose(1, 2)

        # output size [batch_size, length, arc_space]
        arc_h = F.elu(self.arc_h(input))
        arc_c = F.elu(self.arc_c(input))

        # output size [batch_size, length, arc_tag_space]
        arc_tag_h = F.elu(self.arc_tag_h(input))
        arc_tag_c = F.elu(self.arc_tag_c(input))

        # apply dropout
        # [batch_size, length, dim] --> [batch_size, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        arc_tag = torch.cat([arc_tag_h, arc_tag_c], dim=1)

        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)
        arc_tag = self.dropout_out(arc_tag.transpose(1, 2)).transpose(1, 2)

        # output from rnn [batch_size, length, tag_space]
        arc_tag_h, arc_tag_c = arc_tag.chunk(2, 1)
        # head shape [batch_size, length, arc_tag_space]
        arc_tag_h = arc_tag_h.contiguous()
        # child shape [batch_size, length, arc_tag_space]
        arc_tag_c = arc_tag_c.contiguous()
        arc = (arc_h, arc_c)
        # [batch_size, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        out_arc_tag = (arc_tag_h, arc_tag_c)
        return out_arc, out_arc_tag

    def loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None):
        out_arc, out_arc_tag = self.pre_loss(out_arc, out_arc_tag, heads=heads, arc_tags=arc_tags, mask=mask, length=length, use_log=True, temperature=1.0)
        batch_size, max_len = out_arc.size()
        # loss_arc shape [length-1, batch_size]
        out_arc = out_arc.t()
        # loss_arc_tag shape [length-1, batch_size]
        out_arc_tag = out_arc_tag.t()
        # number of valid positions which contribute to loss (remove the symbolic head for each sentence).
        num = mask.sum() - batch_size if mask is not None else float(max_len) * batch_size
        dp_loss = -out_arc.sum() / num, -out_arc_tag.sum() / num
        return dp_loss

    def decode(self, out_arc, out_arc_tag, mask, length, leading_symbolic):
        if self.arc_decode == 'mst':
            heads, arc_tags, scores = self.decode_mst(out_arc, out_arc_tag, mask, length, leading_symbolic)
        else: #self.arc_decode == 'greedy'
            heads, arc_tags, scores = self.decode_greedy(out_arc, out_arc_tag, mask, leading_symbolic)
        return heads, arc_tags, scores

    def decode_mst(self, out_arc, out_arc_tag, mask, length, leading_symbolic):
        loss_arc, loss_arc_tag = self.pre_loss(out_arc, out_arc_tag, heads=None, arc_tags=None, mask=mask, length=length, use_log=True, temperature=1.0)
        batch_size, max_len, _ = loss_arc.size()
        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch_size)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()
        # energy shape [batch_size, num_arcs, length, length]
        energy = torch.exp(loss_arc.unsqueeze(1) + loss_arc_tag)
        heads, arc_tags = parse.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic,
                                           labeled=True)
        heads = from_numpy(heads)
        arc_tags = from_numpy(arc_tags)

        # compute the average score for each tree
        batch_size, max_len = heads.size()
        scores = torch.zeros_like(heads, dtype=energy.dtype, device=energy.device)
        for b_idx in range(batch_size):
            for len_idx in range(max_len):
                scores[b_idx, len_idx] = energy[b_idx, arc_tags[b_idx, len_idx], heads[b_idx, len_idx], len_idx]
        if mask is not None:
            scores = scores.sum(1) / mask.sum(1)
        else:
            scores = scores.sum(1) / max_len
        return heads, arc_tags, scores

    def decode_greedy(self, out_arc, out_arc_tag, mask, leading_symbolic):
        '''
        Args:
            out_arc: Tensor
                the arc scores with shape [batch_size, length, length]
            out_arc_tag: Tensor
                the labeled arc scores with shape [batch_size, length, arc_tag_space]
            mask: Tensor or None
                the mask tensor with shape = [batch_size, length]
            length: Tensor or None
                the length tensor with shape = [batch_size]
            leading_symbolic: int
                number of symbolic labels leading in arc_tag alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and arc_tags.
        '''
        def _decode_arc_tags(out_arc_tag, heads, leading_symbolic):
            # out_arc_tag shape [batch_size, length, arc_tag_space]
            arc_tag_h, arc_tag_c = out_arc_tag
            batch_size, max_len, _ = arc_tag_h.size()
            # create batch index [batch_size]
            batch_index = torch.arange(0, batch_size).type_as(arc_tag_h.data).long()
            # get vector for heads [batch_size, length, arc_tag_space],
            arc_tag_h = arc_tag_h[batch_index, heads.t()].transpose(0, 1).contiguous()
            # compute output for arc_tag [batch_size, length, num_arcs]
            out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)
            # remove the first #leading_symbolic arc_tags.
            out_arc_tag = out_arc_tag[:, :, leading_symbolic:]
            # compute the prediction of arc_tags [batch_size, length]
            _, arc_tags = out_arc_tag.max(dim=2)
            return arc_tags + leading_symbolic

        # out_arc shape [batch_size, length, length]
        out_arc = out_arc.data
        _, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask.data).byte().view(batch_size, max_len, 1)
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch_size, length]
        scores, heads = out_arc.max(dim=1)

        arc_tags = _decode_arc_tags(out_arc_tag, heads, leading_symbolic)

        # compute the average score for each tree
        if mask is not None:
            scores = scores.sum(1) / mask.sum(1)
        else:
            scores = scores.sum(1) / max_len
        return heads, arc_tags, scores

    def pre_loss(self, out_arc, out_arc_tag, heads=None, arc_tags=None, mask=None, length=None, use_log=True, temperature=1.0):
        if (heads is not None and arc_tags is None) or (heads is None and arc_tags is not None):
            raise ValueError('heads and arc_tags should be both Nones or both not Nones')
        decode = True if (heads is None and arc_tags is None) else False
        softmax_func = F.log_softmax if use_log else F.softmax
        # out_arc shape [batch_size, length, length]
        # out_arc_tag shape [batch_size, length, arc_tag_space]
        arc_tag_h, arc_tag_c = out_arc_tag
        batch_size, max_len, arc_tag_space = arc_tag_h.size()
        batch_index = None
        if not decode:
            if length is not None and heads.size(1) != max_len:
                heads = heads[:, :max_len]
                arc_tags = arc_tags[:, :max_len]
            # create batch index [batch_size]
            batch_index = torch.arange(0, batch_size).type_as(out_arc.data).long()
            # get vector for heads [batch_size, length, arc_tag_space],
            arc_tag_h = arc_tag_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        else:
            arc_tag_h = arc_tag_h.unsqueeze(2).expand(batch_size, max_len, max_len, arc_tag_space).contiguous()
            arc_tag_c = arc_tag_c.unsqueeze(1).expand(batch_size, max_len, max_len, arc_tag_space).contiguous()

        # compute output for arc_tag [batch_size, length, num_arcs]
        out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)

        # mask invalid position to -inf for softmax_func
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if not decode:
            # loss_arc shape [batch_size, length, length]
            out_arc = softmax_func(out_arc / temperature, dim=1)
            # loss_arc_tag shape [batch_size, length, num_arcs]
            out_arc_tag = softmax_func(out_arc_tag / temperature, dim=2)
            # mask invalid position to 0 for sum loss
            if mask is not None:
                out_arc = out_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
                out_arc_tag = out_arc_tag * mask.unsqueeze(2)

            # first create index matrix [length, batch_size]
            child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch_size)
            child_index = child_index.type_as(out_arc.data).long()
            # loss_arc shape [batch_size, length-1]
            out_arc = out_arc[batch_index, heads.data.t(), child_index][1:].t()
            # loss_arc_tag shape [batch_size, length-1]
            out_arc_tag = out_arc_tag[batch_index, child_index, arc_tags.data.t()][1:].t()
        else:
            # loss_arc shape [batch_size, length, length]
            out_arc = softmax_func(out_arc / temperature, dim=1)
            # loss_arc_tag shape [batch_size, length, length, num_arcs]
            out_arc_tag = softmax_func(out_arc_tag / temperature, dim=3).permute(0, 3, 1, 2)
        return out_arc, out_arc_tag

class BiRecurrentConv_Encoder(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, use_pos, use_char, pos_dim, num_pos, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, embedd_word=None, embedd_char=None, embedd_pos=None,
                 p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), initializer=None):
        super(BiRecurrentConv_Encoder, self).__init__()
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char) if use_char else None
        self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos) if use_pos else None
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if use_char else None
        self.char_lstm = nn.LSTM(30, 100, num_layers=1, bidirectional=True) if use_char else None
        self.dropout_char_lstm = nn.Dropout(0.5)
        # dropout word
        self.dropout_in = nn.Dropout2d(p_in)
        # standard dropout
        self.dropout_out = nn.Dropout2d(p_out)
        self.dropout_rnn_in = nn.Dropout(p_rnn[0])
        self.use_pos = use_pos
        self.use_char = use_char
        self.rnn_mode = rnn_mode
        self.dim_enc = word_dim
        if use_pos:
            self.dim_enc += pos_dim
        if use_char:
            self.dim_enc += num_filters
        ########################################################
        modelB  = LM_LSTM_CRF([6, 10, 11, 24, 35], 52, 30, 100, 1 ,200 ,512 ,1, 70995, 0.5, 5, True, False, 70995, 1)
        modelB_checkpoint = torch.load('./utils/morph_tagger/cwlm_lstm_crf_cas_2'+'.model',
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        modelB.load_state_dict(modelB_checkpoint['state_dict'])
        self.word_lstms_list = modelB.forward_new()
        ###########################################################

        if rnn_mode == 'RNN':
            RNN = nn.RNN
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
            drop_p_rnn = p_rnn[1]
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        ##########################################################
        self.rnn = RNN(512, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                       dropout=drop_p_rnn)
        ##########################################################
        self.initializer = initializer
        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return

        for name, parameter in self.named_parameters():
            if name.find('embedd') == -1:
                if parameter.dim() == 1:
                    parameter.data.zero_()
                else:
                    self.initializer(parameter.data)

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()
        # pdb.set_trace()
        # [batch_size, length, word_dim]
        word = self.word_embedd(input_word)
        # apply dropout on input
        word = self.dropout_in(word)
        # [1,30,200]
        input = word
        if self.use_char:
            # [batch_size, length, char_length, char_dim]
            #[1,30,16,30]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # pdb.set_trace()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            # char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3])
            # [30,16,30]
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            # char, _ = self.conv1d(char).max(dim=2)
            self.char_lstm.flatten_parameters()
            char, char_hidden = self.char_lstm(char)
            char = self.dropout_char_lstm(char)
            backword_char = char[:,0,100:].view(char_size[0], char_size[1], -1)
            temp_forward = backword_char.clone()
            forward_char = char.view(char_size[0], char_size[1], char_size[2],-1)
            y =torch.ones(input_char.size(),dtype=torch.long).cuda()
            char_end =torch.sum(input_char!=y, dim=2)-1
            char_end[char_end < 0] = 0
            for b in range(char_size[0]):
                for l in range(char_size[1]):
                    temp_forward[b,l,:] = forward_char[b,l,char_end[b,l],:100]
            # Char.size = [30, 16, 200]
            # char_hidden[0] =[2,16,100]
            # reshape to [batch_size, length, char_filters]
            # char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            # char = self.dropout_in(char)
            # concatenate word and char [batch_size, length, word_dim+char_filter]

            input = torch.cat([input, temp_forward,backword_char], dim=2)

        if self.use_pos:
            # [batch_size, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            input = torch.cat([input, pos], dim=2)

        # apply dropout rnn input
        input = self.dropout_rnn_in(input)
        #########################################################3
        prev_lstm_output = None
        lstm_input = input

        for i in range(3):
            if i > 0:
                lstm_input = torch.cat([input, prev_lstm_output], dim=2)
            lstm_output,_ = self.word_lstms_list[i](lstm_input)
            prev_lstm_output = lstm_output

        #word level lstm
        lstm_out = prev_lstm_output
        # d_lstm_out = self.dropout(lstm_out)
        input = lstm_out
        ########################################################
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            self.rnn.flatten_parameters()
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch_size, length, hidden_size]
            self.rnn.flatten_parameters()
            output, hn = self.rnn(input, hx=hx)
        # apply dropout for the output of rnn
        output = self.dropout_out(output)
        return output, hn, mask, length

class Gating(nn.Module):
    # Implementation of:
    # Sato, Motoki, et al. "Adversarial training for cross-domain universal dependency parsing."
    # Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal
    #  Dependencies. 2017.‏
    def __init__(self, num_gates, input_dim):
        super(Gating, self).__init__()
        self.num_gates = num_gates
        self.input_dim = input_dim
        if self.num_gates == 2:
            self.linear = nn.Linear(self.num_gates * self.input_dim, self.input_dim)
        elif self.num_gates > 2:
            self.linear = nn.Linear(self.num_gates * self.input_dim, self.num_gates * self.input_dim)
            self.softmax = nn.Softmax(-1)
        else:
            raise ValueError('num_gates should be greater or equal to 2')

    def forward(self, tuple_of_inputs):
        # output size should be equal to the input sizes
        if self.num_gates == 2:
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            output = torch.mul(alpha, tuple_of_inputs[0]) + torch.mul(1 - alpha, tuple_of_inputs[1])
        else: # elif self.num_gates > 2:
            # extend the gating mechanism to more than 2 encoders
            batch_size, len_size, dim_size = tuple_of_inputs[0].size()
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            alpha = self.softmax(alpha.view(batch_size, len_size, dim_size, self.num_gates))
            output = torch.sum(torch.mul(alpha,torch.stack(tuple_of_inputs, dim=-1)), dim=-1)
        return output

class LM_LSTM_CRF(nn.Module):
    """LM_LSTM_CRF model

    args:
        tagset_sizes: sizes of label set (one for each task/dataset)
        char_size: size of char dictionary
        char_dim: size of char embedding
        char_hidden_dim: size of char-level lstm hidden dim
        char_rnn_layers: number of char-level lstm layers
        embedding_dim: size of word embedding
        word_hidden_dim: size of word-level blstm hidden dim
        word_rnn_layers: number of word-level lstm layers
        vocab_size: size of word dictionary
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
        if_highway: use highway layers or not
        in_doc_words: number of words that occurred in the corpus (used for language model prediction)
        highway_layers: number of highway layers
    """

    def __init__(self, tagset_sizes, char_size, char_dim, char_hidden_dim, char_rnn_layers, embedding_dim, word_hidden_dim, word_rnn_layers, vocab_size, dropout_ratio, file_num, large_CRF=True, if_highway = False, in_doc_words = 2, highway_layers = 1):

        super(LM_LSTM_CRF, self).__init__()
        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_size = char_size
        self.word_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.if_highway = if_highway

        self.char_embeds = nn.Embedding(char_size, char_dim)
        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.char_rnn_layers = char_rnn_layers

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.word_lstms_list = nn.ModuleList()

        word_input_size = embedding_dim + char_hidden_dim * 2

        self.word_lstm_task_0 = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)


        self.word_lstms_list.append(self.word_lstm_task_0)

        for i in range(file_num - 1):
            self.word_lstms_list.append(nn.LSTM(word_input_size + word_hidden_dim, word_hidden_dim // 2, num_layers=1, bidirectional=True, dropout=dropout_ratio))


        self.word_rnn_layers = word_rnn_layers

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.tagset_sizes = tagset_sizes
        self.crflist = nn.ModuleList()
        for i in range(file_num):
            if large_CRF:
                self.crflist.append(crf.CRF_L(word_hidden_dim, tagset_sizes[i]))
            else:
                self.crflist.append(crf.CRF_S(word_hidden_dim, tagset_sizes[i]))

        if if_highway:
            self.forw2char = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2char = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.forw2word = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2word = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.fb2char = highway.hw(2 * char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)

        self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
        self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)

        self.batch_size = 1
        self.word_seq_length = 1


    def forward_new(self):
        return self.word_lstms_list