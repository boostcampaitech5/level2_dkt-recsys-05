import torch
import torch.nn as nn
import numpy as np

"""
Reference:
https://arxiv.org/abs/2002.07033
https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch
"""

class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout=0.1):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn) #[batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model) #[batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return self.dropout(x)

class SaintPlus(nn.Module):
    def __init__(self, seq_len, num_layers, d_ffn, d_model, num_heads, n_problems, n_tests, n_categories, n_tags, dropout):
        super(SaintPlus, self).__init__()
        self.d_model = d_model
        self.n_questions = n_problems
        self.num_heads = num_heads

        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.problem_emb = nn.Embedding(n_problems, d_model)
        self.test_emb = nn.Embedding(n_tests, d_model)
        self.category_emb = nn.Embedding(n_categories, d_model)
        self.tag_emb = nn.Embedding(n_tags, d_model)

        self.timelag_emb = nn.Linear(1, d_model, bias=False)
        self.elapsedT_emb = nn.Linear(1, d_model, bias=False)

        self.answerCorr_emb = nn.Embedding(2, d_model)

        self.emb_dense1 = nn.Linear(5*d_model, d_model)
        self.emb_dense2 = nn.Linear(3*d_model, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=d_ffn, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = FFN(d_ffn, d_model, dropout=dropout)
        self.final_layer = nn.Linear(d_model, 1)
    
    def forward(self, category_ids, test_ids, problem_ids, tag_ids, time_lag, elapsed_time, answer_correct):
        device = problem_ids.device
        seq_len = problem_ids.shape[1]

        problem_id_emb = self.problem_emb(problem_ids)
        test_id_emb = self.test_emb(test_ids)
        category_id_emb = self.category_emb(category_ids)
        tag_id_emb = self.tag_emb(tag_ids)

        time_lag = torch.log(time_lag+1)
        time_lag = time_lag.view(-1, 1) # [batch*seq_len, 1]
        time_lag = self.timelag_emb(time_lag) # [batch*seq_len, d_model]
        time_lag = time_lag.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]
        elapsed_time = torch.log(elapsed_time+1)
        elapsed_time = elapsed_time.view(-1, 1) # [batch*seq_len, 1]
        elapsed_time = self.elapsedT_emb(elapsed_time) # [batch*seq_len, d_model]
        elapsed_time = elapsed_time.view(-1, seq_len, self.d_model) # [batch, seq_len, d_model]    

        answer_correct_emb = self.answerCorr_emb(answer_correct)

        encoder_val = torch.cat((problem_id_emb, test_id_emb, category_id_emb, tag_id_emb, time_lag), axis=-1)
        encoder_val = self.emb_dense1(encoder_val)
        decoder_val = torch.cat((time_lag, elapsed_time, answer_correct_emb), axis=-1)
        decoder_val = self.emb_dense2(decoder_val)
        
        pos = torch.arange(seq_len).unsqueeze(0).to(device)
        pos_emb = self.pos_emb(pos)
        encoder_val += pos_emb
        decoder_val += pos_emb

        over_head_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))
        over_head_mask = over_head_mask.to(device)

        encoder_val = encoder_val.permute(1, 0, 2)
        decoder_val = decoder_val.permute(1, 0, 2)
        decoder_val = self.transformer(encoder_val, decoder_val, src_mask=over_head_mask, tgt_mask=over_head_mask, memory_mask=over_head_mask)

        decoder_val = self.layer_norm(decoder_val)
        decoder_val = decoder_val.permute(1, 0, 2)
        
        final_out = self.FFN(decoder_val)
        final_out = self.layer_norm(final_out + decoder_val)
        final_out = self.final_layer(final_out)
        final_out = torch.sigmoid(final_out)
        return final_out.squeeze(-1)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

if __name__ == "__main__":
    model = SaintPlus(100, 4, 128*4, 128, 8, 1000, 13523, 7, 10000, 4)
    print(model)