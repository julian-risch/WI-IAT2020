from torch import nn
import torch.nn.functional as F
import torch

from src.model.utils import FM
from src.utils.utils import create_embedding_layer


class DeepCoNN(nn.Module):
    def __init__(self,
                 max_length_user_rep,
                 max_length_comment_section,
                 dropout=0.5,
                 user_num_kernels=50,  # number of kernels
                 section_num_kernels=50,
                 user_kernel_size=5,  # number of words in window
                 section_kernel_size=5,
                 user_latent_factors1=16,  # embedding size
                 section_latent_factors1=16,
                 freeze_embeddings=True):
        super(DeepCoNN, self).__init__()
        self.max_length_user_rep = max_length_user_rep
        self.max_length_comment_section = max_length_comment_section

        self.embedding_dim, self.max_length, self.embedding = create_embedding_layer(freeze_embeddings)

        self.user_kernel_size = user_kernel_size
        self.section_kernel_size = section_kernel_size

        self.userCNN = nn.Sequential(
            # convolutional
            nn.Conv2d(1, user_num_kernels, (user_kernel_size, self.embedding_dim), stride=1),
            nn.ReLU(),
        )
        self.sectionCNN = nn.Sequential(
            nn.Conv2d(1, section_num_kernels, (section_kernel_size, self.embedding_dim), stride=1),
            nn.ReLU()
        )

        self.user_linear = nn.Sequential(
            nn.Linear(user_num_kernels, user_latent_factors1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.section_linear = nn.Sequential(
            nn.Linear(section_num_kernels, section_latent_factors1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fm = FM(n=user_latent_factors1 + section_latent_factors1,
                     k=user_latent_factors1 + section_latent_factors1)

    def forward(self, user_content, section_content):
        return self.pred(user_content, section_content)

    def pred(self, user_content, section_content):
        ## look-up content
        user_x = self.embedding(user_content).float()
        section_x = self.embedding(section_content).float()
        # user
        user_x = self.userCNN(user_x)
        user_x = torch.squeeze(user_x, -1)
        user_x = F.max_pool1d(user_x, self.max_length_user_rep - self.user_kernel_size + 1)
        user_x = user_x.squeeze(-1)
        user_x = self.user_linear(user_x)

        # section
        section_x = self.sectionCNN(section_x)
        section_x = torch.squeeze(section_x, -1)
        section_x = F.max_pool1d(section_x, self.max_length_comment_section - self.section_kernel_size + 1)
        section_x = section_x.squeeze(-1)
        section_x = self.section_linear(section_x)

        z = torch.cat((user_x, section_x), 1)
        return torch.squeeze(torch.sigmoid(self.fm(z)), -1)

    def get_user_rep(self, user_content):
        user_x = self.embedding(user_content).float()
        # user
        user_x = self.userCNN(user_x)
        user_x = torch.squeeze(user_x, -1)
        user_x = F.max_pool1d(user_x, self.max_length_user_rep - self.user_kernel_size + 1)
        user_x = user_x.squeeze(-1)
        user_x = self.user_linear(user_x)
        return user_x

    def get_sect_emb(self, section_content):
        section_x = self.embedding(section_content).float()
        section_x = self.sectionCNN(section_x)
        section_x = torch.squeeze(section_x, -1)
        section_x = F.max_pool1d(section_x, self.max_length_comment_section - self.section_kernel_size + 1)
        section_x = section_x.squeeze(-1)
        section_x = self.section_linear(section_x)
        return section_x
