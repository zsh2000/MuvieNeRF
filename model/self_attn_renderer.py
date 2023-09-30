# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # Copyright 2020 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(stdv, stdv)


def masked_softmax(x, mask, **kwargs):
    x_masked = x.masked_fill(mask == 0, -float("inf"))

    return torch.softmax(x_masked, **kwargs)


## Auto-encoder network
class ConvAutoEncoder(nn.Module):
    def __init__(self, num_ch, S):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = masked_softmax(attn, mask, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.transpose(1, 2).unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


#class FixKV_MultiHeadAttention(nn.Module):
#    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#        super().__init__()
#
#        self.n_head = n_head
#        self.d_k = d_k
#        self.d_v = d_v
#
#        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
##        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)        # 4, 32, 8, 8
##        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
#
#        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
#
#        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#
#    def forward(self, q, mapped_k, mapped_v, mask=None):
#
#        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), mapped_k.size(1), mapped_v.size(1)
#
#        residual = q
#
#        # Pass through the pre-attention projection: b x lq x (n*dv)
#        # Separate different heads: b x lq x n x dv
#        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#        k = mapped_k.view(sz_b, len_k, n_head, d_k)
#        v = mapped_v.view(sz_b, len_v, n_head, d_v)
#
#        # Transpose for attention dot product: b x n x lq x dv
#        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#        if mask is not None:
#            mask = mask.transpose(1, 2).unsqueeze(1)  # For head axis broadcasting.
#
#        q, attn = self.attention(q, k, v, mask=mask)
#
#        # Transpose to move the head dimension back: b x lq x n x dv
#        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#        q = self.fc(q)
#        q += residual
#
#        q = self.layer_norm(q)
#
#        return q, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Renderer(nn.Module):
    def __init__(self, nb_samples_per_ray):
        super(Renderer, self).__init__()

        self.dim = 32
        self.attn_token_gen = nn.Linear(24 + 1 + 8, self.dim)

        ## Self-Attention Settings
        d_inner = self.dim
        n_head = 4
        d_k = self.dim // n_head
        d_v = self.dim // n_head
        num_layers = 4
        self.num_tasks = 5
        self.attn_layers = nn.ModuleList(
            [
                EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )

        ## Processing the mean and variance of input features
        self.var_mean_fc1 = nn.Linear(16, self.dim)
        self.var_mean_fc2 = nn.Linear(self.dim, self.dim)

        ## Setting mask of var_mean always enabled
        self.var_mean_mask = torch.tensor([1]).to(device)
        self.var_mean_mask.requires_grad = False

        ## For aggregating data along ray samples
        self.auto_enc = ConvAutoEncoder(self.dim, nb_samples_per_ray)

        self.sigma_fc1 = nn.Linear(self.dim, self.dim)
        self.sigma_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.sigma_fc3 = nn.Linear(self.dim // 2, 1)

        self.rgb_fc1 = nn.Linear(self.dim, self.dim)
#        self.rgb_fc1_2 = nn.Linear(self.dim, self.dim)
        self.rgb_fc2 = nn.Linear(self.dim*2, self.dim)
#        self.rgb_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)
        self.rgb_fc3 = nn.Linear(self.dim, 1)

        self.sn_fc1 = nn.Linear(self.dim, self.dim)
        self.sh_fc1 = nn.Linear(self.dim, self.dim)
        self.kp_fc1 = nn.Linear(self.dim, self.dim)
        self.ed_fc1 = nn.Linear(self.dim, self.dim)

#        self.sn_fc1_2 = nn.Linear(self.dim, self.dim)
#        self.sh_fc1_2 = nn.Linear(self.dim, self.dim)
#        self.kp_fc1_2 = nn.Linear(self.dim, self.dim)
#        self.ed_fc1_2 = nn.Linear(self.dim, self.dim)

        self.sn_fc2 = nn.Linear(self.dim*2, self.dim)
        self.sh_fc2 = nn.Linear(self.dim*2, self.dim)
        self.kp_fc2 = nn.Linear(self.dim*2, self.dim)
        self.ed_fc2 = nn.Linear(self.dim*2, self.dim)

#        self.sn_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)
#        self.sh_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)
#        self.kp_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)
#        self.ed_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)

        self.sn_fc3 = nn.Linear(self.dim, 1)
        self.sh_fc3 = nn.Linear(self.dim, 1)
        self.kp_fc3 = nn.Linear(self.dim, 1)
        self.ed_fc3 = nn.Linear(self.dim, 1)

        self.seg_fc1 = nn.Linear(self.dim, self.dim)
#        self.seg_fc1_2 = nn.Linear(self.dim, self.dim)
        self.seg_fc2 = nn.Linear(self.dim*2, self.dim)
#        self.seg_fc2_2 = nn.Linear(self.dim // 2, self.dim // 2)
        self.seg_fc3 = nn.Linear(self.dim, 1)

        self.cross_attn_rgb_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
        self.cross_attn_sn_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
        self.cross_attn_sh_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
        self.cross_attn_ed_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
        self.cross_attn_kp_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
        self.cross_attn_seg_1 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)

#        self.w_ks = nn.Linear(32, 32, bias=False)
#        self.w_vs = nn.Linear(32, 32, bias=False)

        self.cross_attn_1 = MultiHeadAttention(n_head, 64, 64//n_head, 64//n_head)
        self.cross_attn_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_rgb_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_sn_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_sh_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_ed_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_kp_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#        self.cross_attn_seg_1_2 = MultiHeadAttention(n_head, 32, 32//n_head, 32//n_head)
#
#        self.cross_attn_rgb_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_sn_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_sh_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_ed_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_kp_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_seg_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#
#        self.cross_attn_rgb_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_sn_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_sh_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_ed_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_kp_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)
#        self.cross_attn_seg_2_2 = MultiHeadAttention(n_head, 16, 16//n_head, 16//n_head)



#        self.mha_1 = MultiHeadAttention(n_head, self.dim, 32//n_head, 32//n_head)
#        self.mha_2 = MultiHeadAttention(n_head, self.dim, 32//n_head, 32//n_head)
#        self.mha_3 = MultiHeadAttention(n_head, self.dim + 9, 32//n_head, 32//n_head)
#        self.mha_4 = MultiHeadAttention(n_head, self.dim, 32//n_head, 32//n_head)


        self.mha_1 = EncoderLayer(self.dim + 9, self.dim, n_head, 32//n_head, 32//n_head)
        self.mha_2 = EncoderLayer(self.dim, self.dim, n_head, 32//n_head, 32//n_head)
        self.mha_3 = EncoderLayer(self.dim, self.dim, n_head, 32//n_head, 32//n_head)
        self.mha_4 = EncoderLayer(self.dim, self.dim, n_head, 32//n_head, 32//n_head)








#        self.final_mha = MultiHeadAttention(n_head, self.dim // 2 + 9, 32//n_head, 32//n_head)


#        self.task_prompts = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim + 9))
        self.task_prompts_1 = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim))
#        self.task_prompts_1_2 = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim))
#        self.task_prompts_2 = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim // 2))
#        self.task_prompts_2_2 = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim // 2))



#        self.task_prompts_3 = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim + 9))
#        self.final_task_prompts = nn.Parameter(torch.randn(self.num_tasks + 1, self.dim // 2 + 9))

#        self.projects_prompt_rgb = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )
#
#        self.projects_prompt_sn = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )
#
#
#
#        self.projects_prompt_sh = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )
#
#
#        self.projects_prompt_kp = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )
#
#
#        self.projects_prompt_ed = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )
#
#
#        self.projects_prompt_seg = nn.ModuleList(
#                [nn.Sequential(
#                    nn.Linear(self.dim + 9, self.dim),
#                    nn.ReLU(),
#                    nn.Linear(self.dim, self.dim // 2),
#                )
#                for i in range(4)]
#        )


#        self.q_fc_0 = nn.Sequential(
#            nn.Linear(self.dim + 9, self.dim),
#            nn.ReLU(),
#            nn.Linear(self.dim, self.dim),
#        )
#
        self.q_fc_1 = nn.Sequential(
            nn.Linear(self.dim+9, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )


        self.q_fc_2 = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )


        self.q_fc_3 = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        self.q_fc_4 = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )



        self.sigma_fc3.apply(weights_init)


    def forward(self, viewdirs, feat, occ_masks):
        ## Viewing samples regardless of batch or ray
        N, S, V = feat.shape[:3]
        feat = feat.view(-1, *feat.shape[2:])
        v_feat = feat[..., :24]
        s_feat = feat[..., 24 : 24 + 8]
#        print(feat.shape, "total_shape")
        colors = feat[..., 24 + 8 : 24 + 8 + 3]
        segs = feat[..., 24 + 8 + 3: 24 + 8 + 4]
        sns = feat[..., 24 + 8 + 4: 24 + 8 + 7]
        shs = feat[..., 24 + 8 + 7: 24 + 8 + 8]
        eds = feat[..., 24 + 8 + 8: 24 + 8 + 9]
        kps = feat[..., 24 + 8 + 9: -1]
        vis_mask = feat[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])
        viewdirs = viewdirs.view(-1, *viewdirs.shape[2:])

        ## Mean and variance of 2D features provide view-independent tokens
        var_mean = torch.var_mean(s_feat, dim=1, unbiased=False, keepdim=True)
        var_mean = torch.cat(var_mean, dim=-1)
        var_mean = F.elu(self.var_mean_fc1(var_mean))
        var_mean = F.elu(self.var_mean_fc2(var_mean))

#        print(v_feat)

        ## Converting the input features to tokens (view-dependent) before self-attention
        tokens = F.elu(
            self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
        )
#        tokens = torch.cat([v_feat, vis_mask, s_feat], dim=-1)
#        print(feat)
        tokens = torch.cat([tokens, var_mean], dim=1)

        ## Adding a new channel to mask for var_mean
        vis_mask = torch.cat(
            [vis_mask.to(device), self.var_mean_mask.view(1, 1, 1).expand(N * S, -1, -1).to(device)], dim=1
        )
        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask[:, :-1] *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, masks)

        ## Predicting sigma with an Auto-Encoder and MLP
        sigma_tokens = tokens[:, -1:]
        sigma_tokens = sigma_tokens.view(N, S, self.dim).transpose(1, 2)
        sigma_tokens = self.auto_enc(sigma_tokens)
        sigma_tokens = sigma_tokens.transpose(1, 2).reshape(N * S, 1, self.dim)

      
        rgb_tokens_0_pre_fced = torch.cat([tokens[:, :-1], viewdirs], dim=-1)

#        rgb_tokens_pre_4 = tokens[:, :-1]

#        print(tokens)
#        rgb_tokens_0_pre_fced = self.q_fc_0(rgb_tokens_pre_0)
#        expanded_task_prompts = torch.tile(self.task_prompts, (rgb_tokens_pre.shape[0], 1, 1))
#        prompts_tokens = torch.cat([expanded_task_prompts, rgb_tokens_pre], dim=1)

#        rgb_tokens_pre_1, _ = self.mha_1(rgb_tokens_0_pre_fced, rgb_tokens_0_pre_fced, rgb_tokens_0_pre_fced)
        rgb_tokens_pre_1, _ = self.mha_1(rgb_tokens_0_pre_fced)
        rgb_tokens_1_pre_fced = self.q_fc_1(rgb_tokens_pre_1)


#        expanded_task_prompts_1 = torch.tile(self.task_prompts_1, (rgb_tokens_pre.shape[0], 1, 1))
##        print(expanded_task_prompts_1.shape, rgb_tokens_1.shape)
#
#        prompts_tokens_1 = torch.cat([expanded_task_prompts_1, rgb_tokens_pre_1], dim=1)
#
#        rgb_tokens_pre_2, _ = self.mha_2(rgb_tokens_1_pre_fced, rgb_tokens_1_pre_fced, rgb_tokens_1_pre_fced)

        rgb_tokens_pre_2, _ = self.mha_2(rgb_tokens_1_pre_fced)
        rgb_tokens_2_pre_fced = self.q_fc_2(rgb_tokens_pre_2)
#        rgb_tokens_2_pre_fced = torch.cat([rgb_tokens_2_pre_fced, viewdirs], dim=-1)


#        rgb_tokens_2_viewed = torch.cat([rgb_tokens_2_fced, viewdirs], dim=-1)
#        expanded_task_prompts_2 = torch.tile(self.task_prompts_2, (rgb_tokens_pre.shape[0], 1, 1))
#        prompts_tokens_2 = torch.cat([expanded_task_prompts_2, rgb_tokens_2_viewed], dim=1)

#        rgb_tokens_pre_3, _ = self.mha_3(rgb_tokens_2_pre_fced, rgb_tokens_2_pre_fced, rgb_tokens_2_pre_fced)
        rgb_tokens_pre_3, _ = self.mha_3(rgb_tokens_2_pre_fced)
#        expanded_task_prompts_3 = torch.tile(self.task_prompts_3, (rgb_tokens_pre.shape[0], 1, 1))
#        prompts_tokens_3 = torch.cat([expanded_task_prompts_3, rgb_tokens_3], dim=1)

        rgb_tokens_3_pre_fced = self.q_fc_3(rgb_tokens_pre_3)

#        rgb_tokens_pre_4, _ = self.mha_4(rgb_tokens_3_pre_fced, rgb_tokens_3_pre_fced, rgb_tokens_3_pre_fced)
        rgb_tokens_pre_4, _ = self.mha_4(rgb_tokens_3_pre_fced)
        rgb_tokens_pre_4 = self.q_fc_4(rgb_tokens_pre_4)


        rgb_tokens_4_input = rgb_tokens_pre_4.reshape(N * S, V, self.dim)

#        print(rgb_tokens_4_input)
#        dim_aligned_prompt_rgb_0 = torch.tile(self.projects_prompt_rgb[0](self.task_prompts[0]), (N * S, V, 1))

#        print(dim_aligned_prompt_rgb_0.shape, rgb_tokens_4_input.shape)
#        dim_aligned_prompt_sn_0 = torch.tile(self.projects_prompt_sn[0](self.task_prompts[1]), (N * S, V, 1))
#        dim_aligned_prompt_sh_0 = torch.tile(self.projects_prompt_sh[0](self.task_prompts[2]), (N * S, V, 1))
#        dim_aligned_prompt_ed_0 = torch.tile(self.projects_prompt_ed[0](self.task_prompts[3]), (N * S, V, 1))
#        dim_aligned_prompt_kp_0 = torch.tile(self.projects_prompt_kp[0](self.task_prompts[4]), (N * S, V, 1))
#        dim_aligned_prompt_seg_0 = torch.tile(self.projects_prompt_seg[0](self.task_prompts[5]), (N * S, V, 1))


#        dim_aligned_prompt_rgb_1 = torch.tile(self.projects_prompt_rgb[1](self.task_prompts_1[0]), (N * S, V, 1))
#        dim_aligned_prompt_sn_1 = torch.tile(self.projects_prompt_sn[1](self.task_prompts_1[1]), (N * S, V, 1))
#        dim_aligned_prompt_sh_1 = torch.tile(self.projects_prompt_sh[1](self.task_prompts_1[2]), (N * S, V, 1))
#        dim_aligned_prompt_ed_1 = torch.tile(self.projects_prompt_ed[1](self.task_prompts_1[3]), (N * S, V, 1))
#        dim_aligned_prompt_kp_1 = torch.tile(self.projects_prompt_kp[1](self.task_prompts_1[4]), (N * S, V, 1))
#        dim_aligned_prompt_seg_1 = torch.tile(self.projects_prompt_seg[1](self.task_prompts_1[5]), (N * S, V, 1))
#
#
#
#        dim_aligned_prompt_rgb_2 = torch.tile(self.projects_prompt_rgb[2](self.task_prompts_2[0]), (N * S, V, 1))
#        dim_aligned_prompt_sn_2 = torch.tile(self.projects_prompt_sn[2](self.task_prompts_2[1]), (N * S, V, 1))
#        dim_aligned_prompt_sh_2 = torch.tile(self.projects_prompt_sh[2](self.task_prompts_2[2]), (N * S, V, 1))
#        dim_aligned_prompt_ed_2 = torch.tile(self.projects_prompt_ed[2](self.task_prompts_2[3]), (N * S, V, 1))
#        dim_aligned_prompt_kp_2 = torch.tile(self.projects_prompt_kp[2](self.task_prompts_2[4]), (N * S, V, 1))
#        dim_aligned_prompt_seg_2 = torch.tile(self.projects_prompt_seg[2](self.task_prompts_2[5]), (N * S, V, 1))
#
#
#        dim_aligned_prompt_rgb_3 = torch.tile(self.projects_prompt_rgb[3](self.task_prompts_3[0]), (N * S, V, 1))
#        dim_aligned_prompt_sn_3 = torch.tile(self.projects_prompt_sn[3](self.task_prompts_3[1]), (N * S, V, 1))
#        dim_aligned_prompt_sh_3 = torch.tile(self.projects_prompt_sh[3](self.task_prompts_3[2]), (N * S, V, 1))
#        dim_aligned_prompt_ed_3 = torch.tile(self.projects_prompt_ed[3](self.task_prompts_3[3]), (N * S, V, 1))
#        dim_aligned_prompt_kp_3 = torch.tile(self.projects_prompt_kp[3](self.task_prompts_3[4]), (N * S, V, 1))
#        dim_aligned_prompt_seg_3 = torch.tile(self.projects_prompt_seg[3](self.task_prompts_3[5]), (N * S, V, 1))


#        skip_connect_task_prompts = torch.tile(self.task_prompts, (rgb_tokens_pre.shape[0], 1, 1))

        rgb_tokens = torch.cat([rgb_tokens_4_input], dim=-1)
        sn_tokens = torch.cat([rgb_tokens_4_input], dim=-1)
        sh_tokens = torch.cat([rgb_tokens_4_input], dim=-1)
        ed_tokens = torch.cat([rgb_tokens_4_input], dim=-1)
        kp_tokens = torch.cat([rgb_tokens_4_input], dim=-1)
        seg_tokens = torch.cat([rgb_tokens_4_input], dim=-1)


        sigma_tokens_1 = F.elu(self.sigma_fc1(sigma_tokens)).to(device)
#        print(rgb_tokens.shape)
        rgb_tokens_1 = F.elu(self.rgb_fc1(rgb_tokens)).to(device)
        sn_tokens_1 = F.elu(self.sn_fc1(sn_tokens)).to(device)
        sh_tokens_1 = F.elu(self.sh_fc1(sh_tokens)).to(device)
        ed_tokens_1 = F.elu(self.ed_fc1(ed_tokens)).to(device)
        kp_tokens_1 = F.elu(self.kp_fc1(kp_tokens)).to(device)
        seg_tokens_1 = F.elu(self.seg_fc1(seg_tokens)).to(device)

#  rgb_tokens: (N*S, V, dim)

#        mapped_k = self.w_ks(self.task_prompts_1)
#        mapped_v = self.w_vs(self.task_prompts_1)

        dim_aligned_prompt_1 = torch.tile(self.task_prompts_1, (N * S, 1, 1))

#        dim_aligned_prompt_k = torch.tile(mapped_k, (N * S, 1, 1))
#        dim_aligned_prompt_v = torch.tile(mapped_v, (N * S, 1, 1))

        rgb_tokens_1_before_concat, _ = self.cross_attn_rgb_1(rgb_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)
        sn_tokens_1_before_concat, _ = self.cross_attn_sn_1(sn_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)
        sh_tokens_1_before_concat, _ = self.cross_attn_sh_1(sh_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)
        ed_tokens_1_before_concat, _ = self.cross_attn_ed_1(ed_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)
        kp_tokens_1_before_concat, _ = self.cross_attn_kp_1(kp_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)
        seg_tokens_1_before_concat, _ = self.cross_attn_seg_1(seg_tokens_1, dim_aligned_prompt_1, dim_aligned_prompt_1)

        prompt_1_multiview = torch.tile(torch.unsqueeze(dim_aligned_prompt_1, dim=-2), (1, 1, V, 1))
        rgb_tokens_1 = torch.cat([prompt_1_multiview[:, 0], rgb_tokens_1_before_concat], dim=-1)
        sn_tokens_1 = torch.cat([prompt_1_multiview[:, 1], sn_tokens_1_before_concat], dim=-1)
        sh_tokens_1 = torch.cat([prompt_1_multiview[:, 2], sh_tokens_1_before_concat], dim=-1)
        ed_tokens_1 = torch.cat([prompt_1_multiview[:, 3], ed_tokens_1_before_concat], dim=-1)
        kp_tokens_1 = torch.cat([prompt_1_multiview[:, 4], kp_tokens_1_before_concat], dim=-1)
        seg_tokens_1 = torch.cat([prompt_1_multiview[:, 5], seg_tokens_1_before_concat], dim=-1)




##        print(self.cross_stitch_1, self.cross_stitch_2)
        sigma_tokens_2_input = sigma_tokens_1
#        sh_tokens_2_input = sh_tokens_1
#        ed_tokens_2_input = ed_tokens_1
#        kp_tokens_2_input = kp_tokens_1
#        rgb_tokens_2_input = rgb_tokens_1
#        sn_tokens_2_input = sn_tokens_1
#        seg_tokens_2_input = seg_tokens_1


#        rgb_tokens_1_2 = F.elu(self.rgb_fc1_2(rgb_tokens_1_2_input)).to(device)
#        sn_tokens_1_2 = F.elu(self.sn_fc1_2(sn_tokens_1_2_input)).to(device)
#        sh_tokens_1_2 = F.elu(self.sh_fc1_2(sh_tokens_1_2_input)).to(device)
#        ed_tokens_1_2 = F.elu(self.ed_fc1_2(ed_tokens_1_2_input)).to(device)
#        kp_tokens_1_2 = F.elu(self.kp_fc1_2(kp_tokens_1_2_input)).to(device)
#        seg_tokens_1_2 = F.elu(self.seg_fc1_2(seg_tokens_1_2_input)).to(device)
#
#
#        dim_aligned_prompt_1_2 = torch.tile(self.task_prompts_1_2, (N * S, 1, 1))
#        rgb_tokens_2_input, _ = self.cross_attn_rgb_1_2(rgb_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)
#        sn_tokens_2_input, _ = self.cross_attn_sn_1_2(sn_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)
#        sh_tokens_2_input, _ = self.cross_attn_sh_1_2(sh_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)
#        ed_tokens_2_input, _ = self.cross_attn_ed_1_2(ed_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)
#        kp_tokens_2_input, _ = self.cross_attn_kp_1_2(kp_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)
#        seg_tokens_2_input, _ = self.cross_attn_seg_1_2(seg_tokens_1_2, dim_aligned_prompt_1_2, dim_aligned_prompt_1_2)





        rgb_tokens_1 = torch.unsqueeze(rgb_tokens_1, dim=2).reshape(-1, 1, 64)
        sn_tokens_1 = torch.unsqueeze(sn_tokens_1, dim=2).reshape(-1, 1, 64)
        seg_tokens_1 = torch.unsqueeze(seg_tokens_1, dim=2).reshape(-1, 1, 64)
        sh_tokens_1 = torch.unsqueeze(sh_tokens_1, dim=2).reshape(-1, 1, 64)
        ed_tokens_1 = torch.unsqueeze(ed_tokens_1, dim=2).reshape(-1, 1, 64)
        kp_tokens_1 = torch.unsqueeze(kp_tokens_1, dim=2).reshape(-1, 1, 64)

        combined_tokens_1 = torch.cat([rgb_tokens_1, sn_tokens_1, sh_tokens_1, ed_tokens_1, kp_tokens_1, seg_tokens_1], dim=1)
        combined_tokens_2_input, attn_map_1 = self.cross_attn_1(combined_tokens_1, combined_tokens_1, combined_tokens_1)

#        print(attn_map_1)

        rgb_tokens_2_input, sn_tokens_2_input, sh_tokens_2_input, ed_tokens_2_input, kp_tokens_2_input, seg_tokens_2_input = torch.split(combined_tokens_2_input, 1, dim=1)
        rgb_tokens_2_input = rgb_tokens_2_input.reshape(-1, V, 64)
        sn_tokens_2_input = sn_tokens_2_input.reshape(-1, V, 64)
        seg_tokens_2_input = seg_tokens_2_input.reshape(-1, V, 64)
        sh_tokens_2_input = sh_tokens_2_input.reshape(-1, V, 64)
        ed_tokens_2_input = ed_tokens_2_input.reshape(-1, V, 64)
        kp_tokens_2_input = kp_tokens_2_input.reshape(-1, V, 64)


        sigma_tokens_2 = F.elu(self.sigma_fc2(sigma_tokens_2_input)).to(device)
        rgb_tokens_2 = F.elu(self.rgb_fc2(rgb_tokens_2_input)).to(device)
        sn_tokens_2 = F.elu(self.sn_fc2(sn_tokens_2_input)).to(device)
        sh_tokens_2 = F.elu(self.sh_fc2(sh_tokens_2_input)).to(device)
        ed_tokens_2 = F.elu(self.ed_fc2(ed_tokens_2_input)).to(device)
        kp_tokens_2 = F.elu(self.kp_fc2(kp_tokens_2_input)).to(device)
        seg_tokens_2 = F.elu(self.seg_fc2(seg_tokens_2_input)).to(device)


        sigma_tokens_3_input = sigma_tokens_2
#        sh_tokens_3_input = sh_tokens_2
#        ed_tokens_3_input = ed_tokens_2
#        kp_tokens_3_input = kp_tokens_2
#        rgb_tokens_3_input = rgb_tokens_2
#        sn_tokens_3_input = sn_tokens_2
#        seg_tokens_3_input = seg_tokens_2



        rgb_tokens_2 = torch.unsqueeze(rgb_tokens_2, dim=2).reshape(-1, 1, 32)
        sn_tokens_2 = torch.unsqueeze(sn_tokens_2, dim=2).reshape(-1, 1, 32)
        seg_tokens_2 = torch.unsqueeze(seg_tokens_2, dim=2).reshape(-1, 1, 32)
        sh_tokens_2 = torch.unsqueeze(sh_tokens_2, dim=2).reshape(-1, 1, 32)
        ed_tokens_2 = torch.unsqueeze(ed_tokens_2, dim=2).reshape(-1, 1, 32)
        kp_tokens_2 = torch.unsqueeze(kp_tokens_2, dim=2).reshape(-1, 1, 32)



        combined_tokens_2 = torch.cat([rgb_tokens_2, sn_tokens_2, sh_tokens_2, ed_tokens_2, kp_tokens_2, seg_tokens_2], dim=1)
        combined_tokens_3_input, attn_map_2 = self.cross_attn_2(combined_tokens_2, combined_tokens_2, combined_tokens_2)
        rgb_tokens_3_input, sn_tokens_3_input, sh_tokens_3_input, ed_tokens_3_input, kp_tokens_3_input, seg_tokens_3_input = torch.split(combined_tokens_3_input, 1, dim=1)
        rgb_tokens_3_input = rgb_tokens_3_input.reshape(-1, V, 32)
        sn_tokens_3_input = sn_tokens_3_input.reshape(-1, V, 32)
        seg_tokens_3_input = seg_tokens_3_input.reshape(-1, V, 32)
        sh_tokens_3_input = sh_tokens_3_input.reshape(-1, V, 32)
        ed_tokens_3_input = ed_tokens_3_input.reshape(-1, V, 32)
        kp_tokens_3_input = kp_tokens_3_input.reshape(-1, V, 32)





#        dim_aligned_prompt_2 = torch.tile(self.task_prompts_2, (N * S, 1, 1))
#
#        rgb_tokens_2_2_input, _ = self.cross_attn_rgb_2(rgb_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#        sn_tokens_2_2_input, _ = self.cross_attn_sn_2(sn_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#        sh_tokens_2_2_input, _ = self.cross_attn_sh_2(sh_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#        ed_tokens_2_2_input, _ = self.cross_attn_ed_2(ed_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#        kp_tokens_2_2_input, _ = self.cross_attn_kp_2(kp_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#        seg_tokens_2_2_input, _ = self.cross_attn_seg_2(seg_tokens_2, dim_aligned_prompt_2, dim_aligned_prompt_2)
#
#
#        rgb_tokens_2_2 = F.elu(self.rgb_fc2_2(rgb_tokens_2_2_input)).to(device)
#        sn_tokens_2_2 = F.elu(self.sn_fc2_2(sn_tokens_2_2_input)).to(device)
#        sh_tokens_2_2 = F.elu(self.sh_fc2_2(sh_tokens_2_2_input)).to(device)
#        ed_tokens_2_2 = F.elu(self.ed_fc2_2(ed_tokens_2_2_input)).to(device)
#        kp_tokens_2_2 = F.elu(self.kp_fc2_2(kp_tokens_2_2_input)).to(device)
#        seg_tokens_2_2 = F.elu(self.seg_fc2_2(seg_tokens_2_2_input)).to(device)
#
#
#
#        dim_aligned_prompt_2_2 = torch.tile(self.task_prompts_2_2, (N * S, 1, 1))
#        rgb_tokens_3_input, _ = self.cross_attn_rgb_2_2(rgb_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)
#        sn_tokens_3_input, _ = self.cross_attn_sn_2_2(sn_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)
#        sh_tokens_3_input, _ = self.cross_attn_sh_2_2(sh_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)
#        ed_tokens_3_input, _ = self.cross_attn_ed_2_2(ed_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)
#        kp_tokens_3_input, _ = self.cross_attn_kp_2_2(kp_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)
#        seg_tokens_3_input, _ = self.cross_attn_seg_2_2(seg_tokens_2_2, dim_aligned_prompt_2_2, dim_aligned_prompt_2_2)





        sigma = torch.relu(self.sigma_fc3(sigma_tokens_3_input[:, 0]))

#        prompt_1_multiview = torch.tile(torch.unsqueeze(dim_aligned_prompt_1, dim=-2), (1, 1, V, 1))

        ## Concatenating positional encodings and predicting RGB weights
        rgb_w = self.rgb_fc3(rgb_tokens_3_input)
        rgb_w = masked_softmax(rgb_w, masks[:, :-1], dim=1)

        rgb = (colors * rgb_w).sum(1)

        sn_w = self.sn_fc3(sn_tokens_3_input)
        sn_w = masked_softmax(sn_w, masks[:, :-1], dim=1)

        sn = (sns * sn_w).sum(1)

        sh_w = self.sh_fc3(sh_tokens_3_input)
        sh_w = masked_softmax(sh_w, masks[:, :-1], dim=1)

        sh = (shs * sh_w).sum(1)

        ed_w = self.ed_fc3(ed_tokens_3_input)
        ed_w = masked_softmax(ed_w, masks[:, :-1], dim=1)

        ed = (eds * ed_w).sum(1)

        kp_w = self.kp_fc3(kp_tokens_3_input)
        kp_w = masked_softmax(kp_w, masks[:, :-1], dim=1)
#        print(kps.shape, kp_w.shape, "seg_shape")
        kp = (kps * kp_w).sum(1)

        seg_w = self.seg_fc3(seg_tokens_3_input)
        seg_w = masked_softmax(seg_w, masks[:, :-1], dim=1)

#        print(segs.shape, segs, "seg_shape")
        segs = segs.long()
        seg_vec = F.one_hot(segs, num_classes=13).squeeze(dim=2)
#        print(seg_vec.shape, seg_w.shape, "seg_shape")
       
        seg = (seg_vec * seg_w).sum(1)
#        print(seg.shape, "seg-shape")
        outputs = torch.cat([rgb, seg, sn, sh, ed, kp, sigma], -1)
        outputs = outputs.reshape(N, S, -1)

        return outputs
