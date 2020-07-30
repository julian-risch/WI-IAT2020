import torch
from torch import nn
import torch.nn.functional as F
from src.model.pointwise.graph_only import CommunityGraphModel


class CommunityGraphModelPairwise(nn.Module):
    def __init__(self,
                 node2vec_embedding,
                 node2vec_embedding_dim):
        super(CommunityGraphModelPairwise, self).__init__()

        self.node2vec_embedding = node2vec_embedding
        self.node2vec_embedding_dim = node2vec_embedding_dim

    def forward(self, user_emb, user_emb_offsets, pos_section, neg_section):
        section_emb_pos, section_emb_offsets_pos = pos_section
        section_emb_neg, section_emb_offsets_neg = neg_section
        s1 = self.pred(user_emb, user_emb_offsets, section_emb_pos, section_emb_offsets_pos)
        s2 = self.pred(user_emb, user_emb_offsets, section_emb_neg, section_emb_offsets_neg)
        out = (s1 > s2).float()
        return out

    def pred(self, user_emb, user_emb_offsets, section_emb, section_emb_offsets):
        # look-up node2vec embeddings
        user_emb_x = self.node2vec_embedding(user_emb, user_emb_offsets).float()
        section_emb_x = self.node2vec_embedding(section_emb, section_emb_offsets).float()

        z = F.cosine_similarity(user_emb_x, section_emb_x)
        return z
