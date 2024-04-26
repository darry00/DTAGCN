import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
from layers.MSGBlock import GraphBlock, Attention_Block
from layers.Conv_Block import ConvEncoder

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class MSGBlock(nn.Module):
    def __init__(self, configs):
        super(MSGBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # self.att0 = Attention_Block(configs.d_model, configs.d_ff,
        #                             n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.convatt = ConvEncoder(configs.d_model)
        # self.convatt = ConvEncoder(configs.d_model, configs.num_kernels)
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.seq_len, configs.d_model, configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha, configs.seq_len+configs.pred_len,
                           configs.node_dim))

    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        A = []
        for i in range(self.k):
            scale = scale_list[i]

            #Gconv
            x, adp = self.gconv[i](x)  # [B,T+S,N]
            A.append(adp)
            # padding
            if (self.seq_len+self.pred_len) % scale != 0:
                length = (((self.seq_len+self.pred_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len+self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // scale, scale, N).contiguous()

        # #for Mul-attetion
        #     out = out.reshape(-1, scale, N)
        #     out = self.norm(self.att0(out))
        #     out = self.gelu(out)
        #     out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)

        # #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

        # for CNN
            out = self.norm(self.convatt(out))
            out = out.reshape(B , -1, N)

            out = out[:, :self.seq_len+self.pred_len, :]
            out = out.reshape(B, -1, N)
            res.append(out)

        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res, A

class SGBlock(nn.Module):
    def __init__(self, configs):
        super(SGBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.convatt = ConvEncoder(configs.d_model)
        # self.convatt = ConvEncoder(configs.d_model, configs.num_kernels)
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.enc_dim = configs.enc_in  # scolar
        # self.enc_dim = configs.enc_in + 4
        self.gconv = GraphBlock(configs.c_out, configs.d_model, configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha, self.enc_dim, configs.node_dim)

    def forward(self, x):
        B, T, N = x.size()  # Batch variate dmodel
        #Gconv
        x, adp = self.gconv(x)  # [B,N,E]
        out = x.unsqueeze(1) # [B,1,N,E]
        # #for Mul-attetion
        #     out = out.reshape(-1, scale, N)
        #     out = self.norm(self.att0(out))
        #     out = self.gelu(out)
        #     out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)

        # #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)
        # for CNN
        out = self.norm(self.convatt(out))
        out = out.reshape(B, -1, N)
        res = out + x
        return res, adp

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding_VA = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)  # [B, N, E]
        self.enc_embedding_TA = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)  # [B, L, E]
        self.class_strategy = configs.class_strategy
        self.sgconv = nn.ModuleList([SGBlock(configs) for _ in range(configs.e_layers)])
        self.msgconv = nn.ModuleList([MSGBlock(configs) for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projector_VA = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector_TA = nn.Linear(configs.d_model, configs.enc_in, bias=True)
        self.fusion = nn.Linear(configs.d_model, configs.enc_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N]

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also include covariates

        # Embedding
        # B L N -> B N E
        enc_out_VA = self.enc_embedding_VA(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        # B L N -> B L E
        enc_out_TA = self.enc_embedding_TA(x_enc, x_mark_enc)
        enc_out_TA = self.predict_linear(enc_out_TA.permute(0, 2, 1)).permute(0, 2, 1) # B L E -> B L+S E


        # B N E -> B N S -> B S N
        dec_out_VA = self.projector_VA(enc_out_VA).permute(0, 2, 1)[:, :, :N]  # filter the covariates


        # B L+S E -> B L+S N -> B S N
        dec_out_TA = self.projector_TA(enc_out_TA)[:, -self.pred_len:, :]

        # fusion
        inp_feats  = dec_out_TA + dec_out_VA
        dec_out = self.fusion(inp_feats)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out[:, -self.pred_len:, :]


