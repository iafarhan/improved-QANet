
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import BiDAFAttention, BiDAFOutput, HighwayEncoder,CharCNN, LayerNorm, PositionalEncoding,ProjCNN,QAEmbedding, RNNEncoder
from causal import CausalSelfAttention,clones
from copy import deepcopy
c = deepcopy
from util import masked_softmax


class QAOut(nn.Module):

    def __init__(self, hidden_size, drop_prob):
        super(QAOut, self).__init__()
        self.proj_start = nn.Linear(2 * hidden_size, 1)
        
        self.proj_end = nn.Linear(2 * hidden_size, 1)

    def forward(self, start_out,end_out,mask):
        # Shapes: (batch_size, seq_len, 1)
        s_logits = self.proj_start(start_out)
        e_logits = self.proj_end(end_out)
       
        # # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(s_logits.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(e_logits.squeeze(), mask, log_softmax=True)
        
        return log_p1, log_p2
        #return log_p1, log_p2

class QANet(nn.Module):
    def __init__(self,word_vectors,char_vectors,hidden_size,drop_prob=0.,n_blks=3):
        super(QANet,self).__init__()
        self.emb = QAEmbedding(word_vectors,char_vectors,hidden_size,drop_prob)
        self.emb_enc = EncoderBlock(n_head=8)

        self.cq_attn = BiDAFAttention(hidden_size=hidden_size,drop_prob=drop_prob)

        self.model_blks= clones(EncoderBlock(n_conv=2),n_blks)
        self.proj_attn = ProjCNN(in_channels=4 * hidden_size,out_channels=hidden_size)
        self.out = QAOut(hidden_size,drop_prob)
    def forward(self,cw_idxs,qw_idxs,cc_idxs,qc_idxs):
        c_blk_size = cw_idxs.shape[1]
        q_blk_size = qw_idxs.shape[1]
        #false where cw_idxs is zero. so, it won't be considered while computing softmax probs.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs 
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        subc_mask = torch.tril(torch.ones(c_blk_size,c_blk_size)).\
                                        view(1,1,c_blk_size,c_blk_size)
        subq_mask = torch.tril(torch.ones(q_blk_size,q_blk_size)).\
                                        view(1,1,q_blk_size,q_blk_size)
        
        subc_mask= subc_mask.to(cw_idxs.device)
        subq_mask = subq_mask.to(qw_idxs.device)
        c_emb = self.emb(cw_idxs,cc_idxs) #batch, sent_len, hidden_size
        q_emb = self.emb(qw_idxs,qc_idxs)
        # print(c_emb.shape,q_emb.shape)
        c_enc = self.emb_enc(c_emb,subc_mask)
        q_enc = self.emb_enc(q_emb,subq_mask)
        # print(c_enc.shape,q_enc.shape)
        # To get contextaware query vectors.
        cq_attn = self.cq_attn(c_enc,q_enc,c_mask,q_mask) # (batch, sent_len, 4 * hidden_size)
        #print(cq_attn.shape)
        # print(cq_attn.shape)
        # print(subc_mask.shape)
        x = self.proj_attn(cq_attn.transpose(1,2)).transpose(1,2)
        m_outs = []
        for i in range(3):
            for model_blk in self.model_blks:
                x = model_blk(x,subc_mask)
            m_outs.append(x)
        M0,M1,M2 = m_outs
        M0M1 = torch.cat([M0,M1],dim=-1)
        M0M2 = torch.cat([M0,M2],dim=-1)
        out = self.out(M0M1,M0M2,c_mask)
        return out


class LayerBlock(nn.Module):
    def __init__(self,d_model,pdrop=0.1):
        super(LayerBlock,self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(pdrop)
    def forward(self,x,layer,conv=False,mask=None):
        """
        layer : conv/self_attn/FF
        """
         
        x_norm = self.layer_norm(x)
        if mask is not None:
            out = layer(x_norm,mask)
        elif conv:
            x_norm = x_norm.transpose(1,2)
            out = layer(x_norm).transpose(1,2)
        else:
            out = layer(x_norm)
        return x + self.dropout(out)

class ConvBlock(nn.Module):
    def __init__(self, n_k=128,n_conv=4,d_model=128,pdrop=0.1):
        super().__init__()
        self.conv = DSConv(n_k,n_k)
        self.layer_block = LayerBlock(d_model,pdrop)

    def forward(self,x):
        return self.layer_block(x,self.conv,conv=True)


class EncoderBlock(nn.Module):
    """
    It is a stack of 
    depthwise separable convolution-layer × # + 
    self-attention-layer + feed-forward-layer
    """
    def __init__(self,k=7,hidden_size=128,
                n_k=128,n_conv=4,n_head=8,
                d_model=128,attn_pdrop=0.1,resid_pdrop=0.1,pdrop=0.1):
        super(EncoderBlock,self).__init__()
        self.conv_blks = clones(ConvBlock(d_model,pdrop),n_conv) 
        
        self.self_attn = CausalSelfAttention(d_model,n_head,attn_pdrop,resid_pdrop)
        self.attn_block = LayerBlock(d_model,pdrop)
        self.pos_enc = PositionalEncoding(d_model,pdrop)
        self.ff = nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                nn.ReLU())
        self.ff_block = LayerBlock(d_model,pdrop)
    def forward(self,x,mask):   
        # print(x.shape)     
        x= self.pos_enc(x)
        for conv_blk in self.conv_blks:
            x = conv_blk(x)            
        x = self.attn_block(x,self.self_attn,mask=mask)
        x = self.ff_block(x,self.ff)
        return x

class DSConv(nn.Module):
    def __init__(self,cin,cout,k=7):
        super(DSConv,self).__init__()
        self.depthwise = nn.Conv1d(in_channels=cin,out_channels=cin,kernel_size=k,padding=k//2,groups=cin)
        self.pointwise = nn.Conv1d(in_channels=cin,out_channels=cout,groups=1,kernel_size=1)

    def forward(self,x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return F.relu(out)
