"""Covariance-based similarity metric for few-shot learning.

Extracted from: CovaMNet (Li et al., AAAI 2019)
"Distribution Consistency Based Covariance Metric Networks for Few-Shot Learning"
"""
import torch
import torch.nn as nn


class CovaBlock(nn.Module):
    """Covariance-based similarity module.
    
    Computes similarity based on covariance matrices of support features,
    capturing distribution-level information rather than just point estimates.
    """
    
    def __init__(self):
        super(CovaBlock, self).__init__()

    def cal_covariance(self, input_features: list) -> list:
        """Calculate covariance matrices for each class.
        
        Args:
            input_features: List of (B, C, h, w) feature tensors, one per class
            
        Returns:
            List of (C, C) covariance matrices
        """
        CovaMatrix_list = []
        for i in range(len(input_features)):
            support_set_sam = input_features[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    def cal_similarity(self, query_features: torch.Tensor, CovaMatrix_list: list) -> torch.Tensor:
        """Calculate similarity between query and class covariance matrices.
        
        Args:
            query_features: (B, C, h, w) query feature maps
            CovaMatrix_list: List of (C, C) covariance matrices
            
        Returns:
            (B, num_classes * h * w) similarity scores
        """
        B, C, h, w = query_features.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = query_features[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).to(query_sam.device)

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.view(1, -1))

        Cova_Sim = torch.cat(Cova_Sim, 0)
        return Cova_Sim

    def forward(self, query_features: torch.Tensor, support_features: list) -> torch.Tensor:
        """Compute covariance-based similarity.
        
        Args:
            query_features: (B, C, h, w) query feature maps
            support_features: List of (B, C, h, w) support features per class
            
        Returns:
            (B, num_classes * h * w) similarity scores
        """
        CovaMatrix_list = self.cal_covariance(support_features)
        Cova_Sim = self.cal_similarity(query_features, CovaMatrix_list)
        return Cova_Sim
