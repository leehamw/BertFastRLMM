from torch.nn import functional as F
import torch.nn as nn
import torch
def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))

def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score

def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output




def mlp_attention_score(linear_query, linear_memory, v, tanh, key, query, mask=None):
    hidden_sum = linear_query[0](query).unsqueeze(2) + \
                 linear_memory[0](key).unsqueeze(1)
    key = tanh(hidden_sum)
    attn = v[0](key).squeeze(-1)  # (batch_size, query_length, memory_length)

    return attn


def step_attention( query, key, value, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)
    # query = query.unsqueeze(-2)  # (batch_size,query_length,query_size) (32,1,256)
    # linear_query = nn.ModuleList([nn.Linear(query.size(-1), query.size(-1), bias=True)
    #                               for _ in range(1)])
    # linear_query = linear_query.cuda()
    # linear_memory = nn.ModuleList([nn.Linear(key.size(-1), query.size(-1), bias=False)
    #                                for _ in range(1)])
    # linear_memory = linear_memory.cuda()
    # v = nn.ModuleList([nn.Linear(query.size(-1), 1, bias=False)
    #                    for _ in range(1)])
    # v = v.cuda()
    # tanh = nn.Tanh()
    # # 获得attention分布
    #
    # score = mlp_attention_score(linear_query, linear_memory, v, tanh, key, query, mem_mask)
    # if mem_mask is None:
    #     # 获得softmax过的weight
    #     norm_score = F.softmax(score, dim=-1)
    # else:
    #     # 获得softmax过的weight
    #     norm_score = prob_normalize(score, mem_mask)
    # linear_forget = nn.ModuleList([nn.Linear(query.size(-1), key.size(-1), bias=False)
    #                                for _ in range(1)])
    # linear_forget = linear_forget.cuda()
    # linear_add = nn.ModuleList([nn.Linear(query.size(-1), key.size(-1), bias=False)
    #                             for _ in range(1)])
    # linear_add = linear_add.cuda()
    # rnn_input_size = query.size(-1) + key.size(-1)
    # hidden_size = query.size(-1)
    # rnn = nn.GRU(input_size=rnn_input_size,
    #              hidden_size=hidden_size,
    #              num_layers=1,
    #              dropout=0.2,
    #              batch_first=True)
    # rnn = rnn.cuda()
    #
    # # start1
    #
    # weighted_context = torch.bmm(norm_score, value)
    # rnn_input = torch.cat([weighted_context, query], dim=-1)
    #
    # # hidden=torch.randn(32,hidden_size)
    # rnn_output, new_hidden = rnn(rnn_input, hidden)
    # query = new_hidden[-1].unsqueeze(1)
    #
    # forget = linear_forget[0](query)
    # forget_weights = F.sigmoid(forget)
    #
    # forget_memory = torch.bmm(norm_score.transpose(1, 2), forget_weights)
    # temp_memory = key * (1 - forget_memory)
    # add = linear_add[0](query)
    # add_weights = F.sigmoid(add)
    # add_memory = torch.bmm(norm_score.transpose(1, 2), add_weights)
    # key = temp_memory + add_memory
    #
    # score = mlp_attention_score(linear_query, linear_memory, v, tanh, key, query, mem_mask)
    # if mem_mask is None:
    #     # 获得softmax过的weight
    #     norm_score = F.softmax(score, dim=-1)
    # else:
    #     # 获得softmax过的weight
    #     norm_score = prob_normalize(score, mem_mask)
    #
    # output = attention_aggregate(value, norm_score)
    # return output.squeeze(-2), norm_score.squeeze(-2)
def augmented_attention(query, key, value, attn_lstm,states,linear_forget,linear_add,query_wq, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    #query (batch,hidden) key,value  (batch,seq_len,n_hidden)
    #score  (batch,1,seq_len)

    for i in range(3):
        #address
      score = dot_attention_score(key, query.unsqueeze(-2))
      if mem_mask is None:
          ##batch,1,seq_len
          norm_score = F.softmax(score, dim=-1)
      else:
          norm_score = prob_normalize(score, mem_mask)

      output = attention_aggregate(value, norm_score)
      #batch,hidden
      rnn_input=torch.cat([query,output.squeeze(-2)],dim=1)
      #num_layer,batch,hidden
      states=attn_lstm(rnn_input,states)
      # attn_pre_out=(h,c)
      #batch,hidden
      new_query=query_wq[i](states[0][-1])
      score = dot_attention_score(key, new_query.unsqueeze(-2))
      if mem_mask is None:
          norm_score = F.softmax(score, dim=-1)
      else:
          norm_score = prob_normalize(score, mem_mask)
      #batch,1,hidden
      output = attention_aggregate(value, norm_score)
      forget=linear_forget[i](new_query)
      forget_weights=F.sigmoid(forget)
      ##batch,seq_len,1 batch,1,hidden
      #batch,seq_len,hidden
      forget_memory=torch.bmm(norm_score.transpose(1, 2),forget_weights.unsqueeze(-2))
      temp_memory = key * (1 - forget_memory)
      add=linear_add[i](new_query)
      add_weights=F.sigmoid(add)
      add_memory = torch.bmm(norm_score.transpose(1, 2), add_weights.unsqueeze(-2))
      key=temp_memory+add_memory
          #output:context vector batch,hidden norm_score batch,seq_len
    return output.squeeze(-2), norm_score.squeeze(-2), key


def augmented_attention(query, key, value, attn_lstm,states,linear_forget,linear_add,query_wq, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    #query (batch,hidden) key,value  (batch,seq_len,n_hidden)
    #score  (batch,1,seq_len)

    for i in range(3):
        #address
      score = dot_attention_score(key, query.unsqueeze(-2))
      if mem_mask is None:
          ##batch,1,seq_len
          norm_score = F.softmax(score, dim=-1)
      else:
          norm_score = prob_normalize(score, mem_mask)

      output = attention_aggregate(value, norm_score)
      #batch,hidden
      rnn_input=torch.cat([query,output.squeeze(-2)],dim=1)
      #num_layer,batch,hidden
      states=attn_lstm(rnn_input,states)
      # attn_pre_out=(h,c)
      #batch,hidden
      new_query=query_wq[i](states[0][-1])
      score = dot_attention_score(key, new_query.unsqueeze(-2))
      if mem_mask is None:
          norm_score = F.softmax(score, dim=-1)
      else:
          norm_score = prob_normalize(score, mem_mask)
      #batch,1,hidden
      output = attention_aggregate(value, norm_score)
      forget=linear_forget[i](new_query)
      forget_weights=F.sigmoid(forget)
      ##batch,seq_len,1 batch,1,hidden
      #batch,seq_len,hidden
      forget_memory=torch.bmm(norm_score.transpose(1, 2),forget_weights.unsqueeze(-2))
      temp_memory = key * (1 - forget_memory)
      add=linear_add[i](new_query)
      add_weights=F.sigmoid(add)
      add_memory = torch.bmm(norm_score.transpose(1, 2), add_weights.unsqueeze(-2))
      key=temp_memory+add_memory
          #output:context vector batch,hidden norm_score batch,seq_len
    return output.squeeze(-2), norm_score.squeeze(-2), key