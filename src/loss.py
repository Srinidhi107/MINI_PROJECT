import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing for imbalanced and noisy vulnerability datasets.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 2.0, label_smoothing: float = 0.05):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        """
        num_classes = logits.size(-1)
        
        # 1. Apply label smoothing to one-hot targets
        with torch.no_grad():
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_targets = one_hot * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
            
        # 2. Compute log probabilities and probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # 3. Compute focal weight: (1 - p_t)^gamma
        p_t = (probs * one_hot).sum(dim=-1, keepdim=True) # [B, 1]
        focal_weight = torch.pow(1.0 - p_t, self.gamma) # [B, 1]
        
        # 4. Class weighting: alpha for positive class, (1 - alpha) for negative class
        alpha_factor = torch.where(targets.unsqueeze(1) == 1, self.alpha, 1.0 - self.alpha)
        
        # 5. Loss computation
        cross_entropy = -(smooth_targets * log_probs).sum(dim=-1, keepdim=True)
        loss = alpha_factor * focal_weight * cross_entropy
        return loss.mean()
