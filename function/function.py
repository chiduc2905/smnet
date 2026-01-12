"""Utility functions: loss, seeding, and visualization."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def seed_func(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    """Softmax cross-entropy loss for few-shot classification.
    
    (User requested name: ContrastiveLoss)
    Mathematically equivalent to: -log(exp(score_target) / sum(exp(scores)))
    """
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, way_num) similarity scores
            targets: (N,) class labels
        """
        log_probs = torch.log_softmax(scores, dim=1)
        loss = -log_probs.gather(1, targets.view(-1, 1)).mean()
        return loss


class MarginContrastiveLoss(nn.Module):
    """CosFace-style margin loss for few-shot classification.
    
    Applies margin penalty to correct class score to enforce clear separation:
        s_y = cos(q, s_y) - m    (correct class)
        s_k = cos(q, s_k)        (wrong classes)
    
    This prevents "draw" situations where all classes have similar scores.
    
    Reference: Wang et al. "CosFace: Large Margin Cosine Loss for Deep Face Recognition" (CVPR 2018)
    
    Args:
        margin: Margin to subtract from correct class score (default: 0.2)
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, way_num) similarity scores
            targets: (N,) class labels
            
        Returns:
            loss: Cross-entropy loss with margin penalty on correct class
        """
        N, way_num = scores.shape
        
        # Create one-hot mask for correct class
        one_hot = torch.zeros_like(scores)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        # Apply margin: subtract m from correct class score
        # s_y = s_y - m, s_k = s_k (unchanged)
        scores_with_margin = scores - one_hot * self.margin
        
        # Standard cross-entropy
        log_probs = torch.log_softmax(scores_with_margin, dim=1)
        loss = -log_probs.gather(1, targets.view(-1, 1)).mean()
        
        return loss


class RelationLoss(nn.Module):
    """MSE loss for Relation Networks.
    
    From: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    
    Relation scores are in [0, 1], target is 1 for correct class, 0 for others.
    """
    
    def __init__(self):
        super(RelationLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, Way) relation scores (should be in [0,1] from sigmoid)
            targets: (N,) class labels
        Returns:
            MSE loss
        """
        N, Way = scores.size()
        
        # Create one-hot targets
        one_hot = torch.zeros(N, Way).to(scores.device)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        # MSE between scores and one-hot targets
        loss = self.mse(scores, one_hot)
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        use_gpu (bool): use gpu or not.
    """
    def __init__(self, num_classes=3, feat_dim=1600, use_gpu=True, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device

        # Use device if provided, otherwise fallback to use_gpu flag
        if self.device:
             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        elif self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Normalize centers to unit sphere to match normalized features
        centers_norm = torch.nn.functional.normalize(self.centers, p=2, dim=1)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers_norm, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, centers_norm.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.device:
            classes = classes.to(self.device)
        elif self.use_gpu:
            classes = classes.cuda()
            
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. 2017.
    
    Args:
        margin (float): margin for triplet loss
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class SiameseContrastiveLoss(nn.Module):
    """True Contrastive Loss with Margin for Siamese Networks.
    
    Reference:
    Koch et al. "Siamese Neural Networks for One-shot Image Recognition" (ICML-W 2015)
    
    Original formulation:
    L = y * D² + (1 - y) * max(0, margin - D)²
    
    Where:
    - D = L1 distance between embeddings (or Euclidean)  
    - y = 1 for same class (similar pairs)
    - y = 0 for different class (dissimilar pairs)
    - margin = minimum distance for dissimilar pairs
    
    For N-way classification, we generate all pairs from the episode data
    (support + query) and compute contrastive loss on them.
    """
    
    def __init__(self, margin=1.0):
        super(SiameseContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Compute contrastive loss on all pairs within the batch.
        
        Args:
            embeddings: (N, D) feature embeddings from encoder
            labels: (N,) class labels
        Returns:
            Contrastive loss
        """
        N = embeddings.size(0)
        
        # Compute pairwise L1 distances
        # Using L1 as in the original Siamese paper
        # dist[i,j] = ||emb_i - emb_j||_1
        dist_matrix = torch.cdist(embeddings, embeddings, p=1)  # (N, N)
        
        # Create label matrix: 1 if same class, 0 if different
        labels = labels.view(-1, 1)
        label_matrix = (labels == labels.T).float()  # (N, N)
        
        # Contrastive loss for all pairs
        # L = y * D² + (1 - y) * max(0, margin - D)²
        positive_loss = label_matrix * dist_matrix.pow(2)
        negative_loss = (1 - label_matrix) * F.relu(self.margin - dist_matrix).pow(2)
        
        # Average over all pairs (excluding diagonal)
        mask = 1 - torch.eye(N, device=embeddings.device)
        loss = (positive_loss + negative_loss) * mask
        loss = loss.sum() / mask.sum()
        
        return loss


# Alias for backward compatibility (but SiameseContrastiveLoss is the correct one)
SiameseLoss = SiameseContrastiveLoss


def plot_confusion_matrix(targets, preds, num_classes=3, save_path=None, class_names=None):
    """
    Plot confusion matrix (IEEE format) - saves as PDF vector.
    
    For 200-episode test with 1-query/class: each row sums to 200.
    
    Args:
        targets: Ground truth labels
        preds: Predicted labels
        num_classes: Number of classes
        save_path: Path to save the figure (without extension, will add .pdf)
        class_names: List of class names (default: ['surface', 'corona', 'nopd'])
    """
    # Default class names
    if class_names is None:
        class_names = ['Corona', 'NotPD', 'Surface', 'Void']
    
    # IEEE format: Times New Roman, 14pt font
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14
    })
    
    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    
    # Save in 2-column IEEE layout only
    width = 7.16  # 2-column: 7.16 inches
    layout_name = '2col'
    if True:  # Keep indentation structure
        # Square figure
        fig, ax = plt.subplots(figsize=(width, width))
        
        # Annotations: count and percentage (12pt)
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
        
        # Green colormap (like pd_cnn)
        sns.heatmap(cm, annot=annot, fmt='', cmap='Greens',
                    linewidths=0.5, linecolor='white', ax=ax,
                    annot_kws={'size': 14},
                    vmin=0, square=True,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'shrink': 0.8})
        
        # No title (IEEE format)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xticklabels(class_names, fontsize=14, rotation=45, ha='right')
        ax.set_yticklabels(class_names, fontsize=14, rotation=0)
        
        # Adjust colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        if save_path:
            # Remove extension if present
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            # Save as PDF (vector for publication)
            pdf_path = f"{base_path}_{layout_name}.pdf"
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
            print(f'Saved: {pdf_path}')
            # Save as PNG (for WandB logging)
            png_path = f"{base_path}_{layout_name}.png"
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            print(f'Saved: {png_path}')
        plt.close()


def plot_tsne(features, labels, num_classes=3, save_path=None, class_names=None, title=None):
    """
    t-SNE visualization - Q1 Publication Quality (IEEE/Nature style).
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding):
    - Focuses on preserving LOCAL structure (nearby points stay nearby)
    - Cluster distances are NOT meaningful
    - Good for visualizing tight clusters
    
    Args:
        features: (N, D) feature matrix
        labels: (N,) class labels (0, 1, 2, ...)
        num_classes: Number of classes
        save_path: Path to save the figure
        class_names: List of class names (if None, uses default)
        title: Optional title for the plot
    """
    # ================================================================
    # Q1 Publication Style (Nature/Science/IEEE)
    # ================================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelweight': 'bold',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

    n = len(features)
    unique_n = len(np.unique(features, axis=0))
    print(f"t-SNE: Plotting {n} points (Unique: {unique_n})")
    
    # 1. StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. PCA pre-processing (reduces noise, speeds up t-SNE)
    n_components = min(30, n, features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print(f"  PCA reduced to {n_components} dimensions")
    
    # 3. t-SNE with optimized parameters
    perp = 20  # Fixed perplexity (optimal for 45-150 points in few-shot)
    tsne = TSNE(
        n_components=2, 
        perplexity=perp, 
        random_state=42, 
        init='pca',
        learning_rate='auto',
        max_iter=1000
    )
    embedded = tsne.fit_transform(features_pca)
    
    # Rescale to [-55, 55] to fit in [-60, 60] with margin
    max_val = np.abs(embedded).max()
    if max_val > 0:
        embedded = embedded / max_val * 55
    
    # ================================================================
    # Figure Setup - Q1 Quality
    # ================================================================
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)  # 5x5 inches, high DPI
    
    # Class names
    default_class_names = ['Corona', 'NotPD', 'Surface']
    if class_names is None:
        class_names = default_class_names
    unique_labels = sorted(set(labels))
    
    # Nature/Science color palette (colorblind-friendly)
    # Based on: https://www.nature.com/documents/NRJs-style-guide.pdf
    nature_colors = [
        '#E64B35',  # Vermillion (red-orange)
        '#4DBBD5',  # Sky Blue
        '#00A087',  # Teal/Green
        '#8E55AA',  # Purple
        '#F39B7F',  # Coral
        '#3C5488',  # Blue
    ]
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        class_name = class_names[i] if i < len(class_names) else str(label)
        color = nature_colors[i % len(nature_colors)]
        
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[color], 
            s=50,  # Marker size
            alpha=0.8,
            marker='o',  # Circle marker only
            edgecolors='white', 
            linewidths=0.6,
            label=class_name,
            zorder=3
        )
    
    # ================================================================
    # Axes and Grid - Clean Q1 Style
    # ================================================================
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_aspect('equal')
    
    # Subtle grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Legend - outside plot for clarity
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.3
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Optional title
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # ================================================================
    # Save in Multiple Formats
    # ================================================================
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # PDF for publication (vector graphics)
        pdf_path = f"{base_path}_tsne.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {pdf_path}')
        
        # PNG for presentations/web (high-res raster)
        png_path = f"{base_path}_tsne.png"
        plt.savefig(png_path, format='png', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {png_path}')
        
        # EPS for LaTeX (some journals prefer this)
        eps_path = f"{base_path}_tsne.eps"
        plt.savefig(eps_path, format='eps', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {eps_path}')
    
    plt.close()


def plot_umap(features, labels, num_classes=3, save_path=None, class_names=None, title=None):
    """
    UMAP visualization - Q1 Publication Quality (IEEE/Nature style).
    
    UMAP (Uniform Manifold Approximation and Projection):
    - Preserves both LOCAL and GLOBAL structure
    - Cluster distances are relatively meaningful
    - Faster than t-SNE, more stable across runs
    - Published: McInnes et al., 2018 (arXiv:1802.03426)
    
    Key differences from t-SNE:
    - t-SNE: Only local structure preserved, distances between clusters meaningless
    - UMAP: Both local + global structure, distances relatively meaningful
    
    Args:
        features: (N, D) feature matrix
        labels: (N,) class labels (0, 1, 2, ...)
        num_classes: Number of classes
        save_path: Path to save the figure
        class_names: List of class names (if None, uses default)
        title: Optional title for the plot
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Run: pip install umap-learn")
        print("Falling back to t-SNE...")
        return plot_tsne(features, labels, num_classes, save_path, class_names, title)
    
    # ================================================================
    # Q1 Publication Style (Nature/Science/IEEE)
    # ================================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelweight': 'bold',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

    n = len(features)
    unique_n = len(np.unique(features, axis=0))
    print(f"UMAP: Plotting {n} points (Unique: {unique_n})")
    
    # 1. StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. UMAP with optimized parameters for publication
    # n_neighbors: 15-50 (higher = more global structure)
    # min_dist: 0.0-0.5 (lower = tighter clusters)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedded = reducer.fit_transform(features_scaled)
    print(f"  UMAP embedding complete")
    
    # Rescale to [-55, 55] to fit in [-60, 60] with margin
    max_val = np.abs(embedded).max()
    if max_val > 0:
        embedded = embedded / max_val * 55
    
    # ================================================================
    # Figure Setup - Q1 Quality
    # ================================================================
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    
    # Class names
    default_class_names = ['Corona', 'NotPD', 'Surface']
    if class_names is None:
        class_names = default_class_names
    unique_labels = sorted(set(labels))
    
    # Nature/Science color palette (colorblind-friendly)
    nature_colors = [
        '#E64B35',  # Vermillion (red-orange)
        '#4DBBD5',  # Sky Blue
        '#00A087',  # Teal/Green
        '#8E55AA',  # Purple
        '#F39B7F',  # Coral
        '#3C5488',  # Blue
    ]
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        class_name = class_names[i] if i < len(class_names) else str(label)
        color = nature_colors[i % len(nature_colors)]
        
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[color], 
            s=50,
            alpha=0.8,
            marker='o',  # Circle marker only
            edgecolors='white', 
            linewidths=0.6,
            label=class_name,
            zorder=3
        )
    
    # ================================================================
    # Axes and Grid - Clean Q1 Style
    # ================================================================
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_aspect('equal')
    
    # Subtle grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Legend
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.3
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Optional title
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # ================================================================
    # Save in Multiple Formats
    # ================================================================
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # PDF for publication
        pdf_path = f"{base_path}_umap.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {pdf_path}')
        
        # PNG for presentations
        png_path = f"{base_path}_umap.png"
        plt.savefig(png_path, format='png', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {png_path}')
        
        # EPS for LaTeX
        eps_path = f"{base_path}_umap.eps"
        plt.savefig(eps_path, format='eps', bbox_inches='tight', 
                    facecolor='white', edgecolor='none', dpi=300)
        print(f'Saved: {eps_path}')
    
    plt.close()



def plot_tsne_comparison(original_features, encoded_features, labels, num_classes=3, save_path=None):
    """
    t-SNE visualization comparing original (raw pixels) vs encoded features side-by-side.
    
    Args:
        original_features: Raw image features flattened (N, H*W*C)
        encoded_features: Features after encoder (N, feat_dim)
        labels: Class labels (N,)
        num_classes: Number of classes
        save_path: Path to save the figure
    """
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    n = len(labels)
    
    # Process original features
    scaler_orig = StandardScaler()
    orig_scaled = scaler_orig.fit_transform(original_features)
    n_comp_orig = min(50, n-1, original_features.shape[1])
    pca_orig = PCA(n_components=n_comp_orig, random_state=42)
    orig_pca = pca_orig.fit_transform(orig_scaled)
    perp = 5  # Fixed perplexity for tighter class clusters
    tsne_orig = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    orig_embedded = tsne_orig.fit_transform(orig_pca)
    
    # Process encoded features
    scaler_enc = StandardScaler()
    enc_scaled = scaler_enc.fit_transform(encoded_features)
    n_comp_enc = min(30, n-1, encoded_features.shape[1])
    pca_enc = PCA(n_components=n_comp_enc, random_state=42)
    enc_pca = pca_enc.fit_transform(enc_scaled)
    tsne_enc = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    enc_embedded = tsne_enc.fit_transform(enc_pca)
    
    # Rescale both to [-45, 45]
    for embedded in [orig_embedded, enc_embedded]:
        max_val = np.abs(embedded).max()
        if max_val > 0:
            embedded[:] = embedded / max_val * 45
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style('white')
    
    # Original (Raw) t-SNE
    ax1 = axes[0]
    sns.scatterplot(
        x=orig_embedded[:, 0], y=orig_embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend=False, ax=ax1
    )
    ax1.set_title(f'Original Data (Raw Pixels)\n{n} samples', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Dim 1', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Dim 2', fontsize=16, fontweight='bold')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax=ax1)
    
    # Encoded t-SNE
    ax2 = axes[1]
    scatter = sns.scatterplot(
        x=enc_embedded[:, 0], y=enc_embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend='full', ax=ax2
    )
    ax2.set_title(f'After Encoder\n{n} samples', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Dim 1', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Dim 2', fontsize=16, fontweight='bold')
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()


def plot_model_comparison_bar(model_results, training_samples, save_path=None):
    """
    Plot horizontal bar chart comparing model performance for 1-shot and 5-shot.
    
    Args:
        model_results: dict with model names as keys and dict {'1shot': acc, '5shot': acc} as values
                      Example: {'CosineNet': {'1shot': 0.9667, '5shot': 0.9800}, ...}
        training_samples: Number of training samples (for title)
        save_path: Path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    # Set font properties
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    models = list(model_results.keys())
    acc_1shot = [model_results[m]['1shot'] * 100 for m in models]
    acc_5shot = [model_results[m]['5shot'] * 100 for m in models]
    
    # Sort by 5-shot accuracy (descending)
    sorted_indices = np.argsort(acc_5shot)[::-1]
    models = [models[i] for i in sorted_indices]
    acc_1shot = [acc_1shot[i] for i in sorted_indices]
    acc_5shot = [acc_5shot[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.8 + 2))
    
    y = np.arange(len(models))
    height = 0.35
    
    # Bars
    bars_5shot = ax.barh(y - height/2, acc_5shot, height, label='5 Shot', color='#5DA5DA', edgecolor='white')
    bars_1shot = ax.barh(y + height/2, acc_1shot, height, label='1 Shot', color='#FAA43A', edgecolor='white')
    
    # Add value labels on bars
    for bar, val in zip(bars_5shot, acc_5shot):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha='left', fontsize=11, color='#5DA5DA', fontweight='bold')
    
    for bar, val in zip(bars_1shot, acc_1shot):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha='left', fontsize=11, color='#FAA43A', fontweight='bold')
    
    # Customize
    ax.set_xlabel('Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    ax.set_title(f'Performance distribution table for the case of {training_samples} samples', 
                 fontsize=18, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlim(50, 100)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    
    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot combined train/val accuracy and loss curves on same figure.
    
    IEEE format: Times New Roman, appropriate figure size.
    
    Args:
        history: dict with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss'
                 each containing list of values per epoch
        save_path: Path to save the figure (without extension, will add .png)
    
    Returns:
        fig: matplotlib figure object
    """
    # IEEE format fonts
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11
    })
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Color scheme
    train_color = '#2E86AB'  # Blue
    val_color = '#E94F37'    # Red
    
    # ===== Accuracy Plot =====
    ax1 = axes[0]
    ax1.plot(epochs, history['train_acc'], color=train_color, 
             linewidth=2, label='Train', marker='o', markersize=3)
    ax1.plot(epochs, history['val_acc'], color=val_color, 
             linewidth=2, linestyle='--', label='Validation', marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # ===== Loss Plot =====
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], color=train_color, 
             linewidth=2, label='Train', marker='o', markersize=3)
    ax2.plot(epochs, history['val_loss'], color=val_color, 
             linewidth=2, linestyle='--', label='Validation', marker='s', markersize=3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        # Save combined figure
        full_path = f"{save_path}_curves.png" if not save_path.endswith('.png') else save_path
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {full_path}')
    
    plt.close()
    return fig