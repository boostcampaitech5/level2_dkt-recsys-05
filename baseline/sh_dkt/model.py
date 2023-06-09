import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import re


try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        # (test, question,tag, correct, mask, interaction, gather_index) - process_batch의 결과
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(
            interaction
        )  # self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)

        embed_test = self.embedding_test(
            test
        )  # self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        embed_question = self.embedding_question(
            question
        )  # nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)

        embed_tag = self.embedding_tag(
            tag
        )  # nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        """
        BertModel의 output을 보면 다음과 같다.

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        즉, out = sequence_output + encoder_outputs[1:] 의 값이 튀어나온다.
        """

        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


from transformers import ElectraModel, ElectraConfig


class Electra(nn.Module):
    def __init__(self, args):
        super(Electra, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.embedding_size = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Electra config
        self.config = ElectraConfig(
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Electra Layer
        self.encoder = ElectraModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Electra
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)

        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.scale * self.pe[: x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.0

        ### Embedding
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim // 3) * 3, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.hidden_dim, self.dropout, self.args.max_seq_len
        )
        self.pos_decoder = PositionalEncoding(
            self.hidden_dim, self.dropout, self.args.max_seq_len
        )

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="relu",
        )

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_enc = torch.cat(
            [
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat(
            [embed_test, embed_question, embed_tag, embed_interaction], 2
        )

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(
            embed_enc,
            embed_dec,
            src_mask=self.enc_mask,
            tgt_mask=self.dec_mask,
            memory_mask=self.enc_dec_mask,
        )

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # Encoder
        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def get_mask(self, seq_len, index, batch_size):
        """
        batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다

        참고로 (batch_size*self.args.n_heads, seq_len, seq_len) 가 아니라
              (batch_size*self.args.n_heads,       1, seq_len) 로 하는 이유는

        last query라 output의 seq부분의 사이즈가 1이기 때문이다
        """
        # [[1], -> [1, 2, 3]
        #  [2],
        #  [3]]
        index = index.view(-1)

        # last query의 index에 해당하는 upper triangular mask의 row를 사용한다
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))
        mask = mask[index]

        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(
            batch_size * self.args.n_heads, -1, seq_len
        )
        return mask.masked_fill(mask == 1, float("-inf"))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, mask, interaction, index = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)

        # 이 3D gathering은 머리가 아픕니다. 잠시 머리를 식히고 옵니다.
        q = torch.gather(q, 1, index.repeat(1, self.hidden_dim).unsqueeze(1))
        q = q.permute(1, 0, 2)

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = self.get_mask(seq_len, index, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )

        self.ffn1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.ffn2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        if self.args.layer_norm:
            self.ln1 = nn.LayerNorm(self.hidden_dim)
            self.ln2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, embed, mask):
        q = self.query(embed).permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        out, _ = self.attn(q, k, v, attn_mask=mask)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out

        if self.args.layer_norm:
            out = self.ln1(out)

        ## feed forward network
        out = self.ffn1(out)
        out = F.relu(out)
        out = self.ffn2(out)

        ## residual + layer norm
        out = embed + out

        if self.args.layer_norm:
            out = self.ln2(out)

        return out


class FixupEncoder(nn.Module):
    def __init__(self, args):
        super(FixupEncoder, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Encoder
        self.encoders = nn.ModuleList(
            [EncoderLayer(args) for _ in range(self.n_layers)]
        )

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

        # T-Fixup
        if self.args.Tfixup:
            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r"^embedding*", name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r".*ln.*|.*bn.*", name):
                continue
            elif re.match(r".*weight*", name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)

    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():
            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r"^embedding*", name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(
                r"encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$", name
            ):
                temp_state_dict[name] = (
                    0.67 * (self.args.n_layers) ** (-1 / 4)
                ) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (
                    param * (2**0.5)
                )

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def mask_2d_to_3d(self, mask, batch_size, seq_len):
        # padding 부분에 1을 주기 위해 0과 1을 뒤집는다
        mask = torch.ones_like(mask) - mask

        mask = mask.repeat(1, seq_len)
        mask = mask.view(batch_size, -1, seq_len)
        mask = mask.repeat(1, self.args.n_heads, 1)
        mask = mask.view(batch_size * self.args.n_heads, -1, seq_len)

        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        embed = self.comb_proj(embed)

        ### Encoder
        mask = self.mask_2d_to_3d(mask, batch_size, seq_len).to(self.device)
        for encoder in self.encoders:
            embed = encoder(embed, mask)

        ###################### DNN #####################
        out = embed.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)
        return preds


from transformers import XLNetModel, XLNetConfig


class XLNet(nn.Module):
    def __init__(self, args):
        super(XLNet, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # XLNet config
        self.config = XLNetConfig(
            3,  # not used
            d_model=self.hidden_dim,
            n_layer=self.args.n_layers,
            n_head=self.args.n_heads,
            total_seq_length=self.args.max_seq_len,
        )

        # Defining the layers
        # XLNet Layer
        self.encoder = XLNetModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, correct, mask, interaction, _ = input
        test = test.flip(dims=[1])
        question = question.flip(dims=[1])
        tag = tag.flip(dims=[1])
        correct = correct.flip(dims=[1])
        mask = mask.flip(dims=[1])
        interaction = interaction.flip(dims=[1])

        batch_size = interaction.size(0)

        # embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag], 2)
        X = self.comb_proj(embed)

        # XLNet
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


from transformers import RobertaModel, RobertaConfig


class Roberta(nn.Module):
    def __init__(self, args):
        super(Roberta, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Roberta config
        self.config = RobertaConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
            pad_token_id=0,  # RoBERTa uses 1 as the default pad token id
            bos_token_id=1,
            eos_token_id=1,
        )

        # Defining the layers
        # Roberta Layer
        self.encoder = RobertaModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Roberta

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)

        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
