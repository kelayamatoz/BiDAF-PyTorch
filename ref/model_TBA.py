import torch
import torch.nn as nn
import layers as L


class BIDAF(nn.Module):
    def __init__(self, args):
        super(BIDAF, self).__init__()

        batch_size, d_hidden = args.batch_size, args.d_hidden
        max_num_sents, max_sent_size = args.max_num_sents, args.max_sent_size
        max_ques_size, max_word_size = args.max_ques_size, args.max_word_size
        word_vocab_size, char_vocab_size = args.word_vocab_size, args.char_vocab_size
        d_char_embed, d_embed = args.d_char_embed, args.d_embed
        d_char_out = args.d_char_out

        seq_in_size = 4*d_hidden
        lin_config = [seq_in_size]*2
        self.char_embed = L.FixedEmbedding(char_vocab_size, d_char_embed)
        self.word_embed = L.FixedEmbedding(word_vocab_size, d_embed)
        self.h_net = L.HighwayNet(d_embed, args.n_hway_layers)
        #self.pre_encoder = L.BiEncoder(word_embed_size, args)
        #self.attend = L.BiAttention(size, args)
        #self.start_encoder0 = L.BiEncoder(word_embed_size, args)
        #self.start_encoder1 = L.BiEncoder(word_embed_size, args)
        #self.end_encoder = L.BiEncoder(word_embed_size, args)
        self.lin_start = L.TFLinear(*lin_config, args.answer_func)
        self.lin_end = L.TFLinear(*lin_config, args.answer_func)

        self.enc_start_shape = (batch_size, max_num_sents * max_sent_size, d_hidden * 2)
        self.logits_reshape = (batch_size, max_num_sents * max_sent_size)
        self.args = args

    def forward(self, ctext, text, text_mask, cquery, query, query_mask):
        a = self.args
 
        # Character Embedding Layer
        ctext_embed = self.char_embed(ctext)
        cquery_embed = self.char_embed(cquery)
        ctext_embed = self.conv(ctext_embed)
        cquery_embed = self.conv(cquery_embed)

        # Word Embedding Layer
        text_embed = self.word_embed(text)
        query_embed = self.word_embed(query)

        # a learned joint character / word embedding
        text = self.h_net(torch.cat((ctext_embed, text_embed), 3))
        query = self.h_net(torch.cat((cquery_embed, query_embed), 2))

        # Contextual Embedding Layer
        text = self.pre_encoder(text)
        query = self.pre_encoder(query)

        # Attention Flow Layer
        text_attn = self.attend(text, query, text_mask, query_mask)

        # The input to the modeling layer is G, which encodes the
        # query-aware representations of context words.
        # Modeling Layer
        text_attn_enc_start = self.start_encoder0(text_attn)
        text_attn_enc_start = self.start_encoder1(text_attn_enc_start)

        # p1 = softmax(w^T_{p1}[G;M])
        # Output Layer
        logits_start = self.lin_start(text_attn_enc_start, text_attn, text_mask)
        start = L.softmax3d(logits_start, a.max_num_sents, a.max_sent_size)

        # softmax of weights from start - not really explained in the paper
        a1i = L.softsel(text_attn_enc_start.view(self.enc_start_shape),
                           logits_start.view(self.logits_reshape))\
                           .unsqueeze(1).unsqueeze(1).repeat(1, a.max_num_sents, a.max_sent_size, 1)

        span = torch.cat((text_attn, text_attn_enc_start, a1i, text_attn_enc_start * a1i), 3)
        text_attn_enc_end = self.end_encoder(span)
        logits_end = self.lin_end(text_attn_enc_end, text_attn, text_mask)
        end = L.softmax3d(logits_end, a.max_num_sents, a.max_sent_size)
        return start, end
