
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

class SimpleEdgeLoss(nn.Module):

    
    def __init__(self, edge_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        self.current_stage = 1
        
        self.laplacian_filter = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.stage_weights = {
            1: {'detail': 1.0, 'structure': 0.0, 'ssim': 0.0}, 
            2: {'detail': 0.6, 'structure': 0.4, 'ssim': 0.3}, 
            3: {'detail': 0.3, 'structure': 0.7, 'ssim': 0.5} 
        }

    
    def update_stage(self, stage):
        assert 1 <= stage <= 3, 
        self.current_stage = stage
        return self
    
    def to_grayscale(self, x):
        if x.size(1) > 1:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x
    
    def extract_edges(self, x):
        device = x.device
        x_gray = self.to_grayscale(x)
        
        laplacian = self.laplacian_filter.to(device)
        edges = F.conv2d(x_gray, laplacian, padding=1)
        return torch.abs(edges)  
    
    def compute_ssim_loss(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y.pow(2), kernel_size=3, stride=1, padding=1) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_xy
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()  
    
    def compute_multiscale_loss(self, pred, target, mask):
        device = pred.device
        
        scales = []
        
        scales.append((pred, target, mask))
        
        if max(pred.shape[2:]) >= 64:
            scale_factor = 0.5
            pred_med = F.interpolate(pred, scale_factor=scale_factor, mode='bilinear')
            target_med = F.interpolate(target, scale_factor=scale_factor, mode='bilinear')
            mask_med = F.interpolate(mask, scale_factor=scale_factor, mode='nearest')
            scales.append((pred_med, target_med, mask_med))
        
        if max(pred.shape[2:]) >= 128:
            scale_factor = 0.25
            pred_large = F.interpolate(pred, scale_factor=scale_factor, mode='bilinear')
            target_large = F.interpolate(target, scale_factor=scale_factor, mode='bilinear')
            mask_large = F.interpolate(mask, scale_factor=scale_factor, mode='nearest')
            scales.append((pred_large, target_large, mask_large))
        
        total_loss = 0
        weights = [1.0, 0.7, 0.5][:len(scales)]  
        
        for i, (p, t, m) in enumerate(scales):
            p_edges = self.extract_edges(p)
            t_edges = self.extract_edges(t)
            
            outpaint_mask = 1.0 - m
            
            if outpaint_mask.sum() > 0:
                loss = F.l1_loss(
                    p_edges * outpaint_mask, 
                    t_edges * outpaint_mask, 
                    reduction='sum'
                ) / (outpaint_mask.sum() + 1e-8)
                
                total_loss += weights[i] * loss
        
        return total_loss / sum(weights)
    
    def forward(self, pred, target, mask=None):
        device = pred.device  
        
        try:
            if pred.size() != target.size():
                pred = F.interpolate(pred, size=target.size()[2:], mode='bilinear')
            
            if mask is None:
                mask = torch.zeros(pred.size(0), 1, pred.size(2), pred.size(3), device=device)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1) 
                mask = mask.to(device) 
            else:
                mask = mask.to(device)
                
            outpaint_mask = 1.0 - mask 
            
            weights = self.stage_weights[self.current_stage]
            
            losses = {}
            total_loss = 0.0
            
            pred_edges = self.extract_edges(pred)
            target_edges = self.extract_edges(target)
            
            if outpaint_mask.sum() > 0:
                detail_loss = F.l1_loss(
                    pred_edges * outpaint_mask,
                    target_edges * outpaint_mask,
                    reduction='sum'
                ) / (outpaint_mask.sum() + 1e-8)
                
                losses['detail'] = detail_loss
                total_loss += weights['detail'] * detail_loss
            
            if weights['structure'] > 0:
                structure_loss = self.compute_multiscale_loss(pred, target, mask)
                losses['structure'] = structure_loss
                total_loss += weights['structure'] * structure_loss
            
            if weights['ssim'] > 0:
                pred_blur = gaussian_blur(pred, kernel_size=5, sigma=1.5)
                target_blur = gaussian_blur(target, kernel_size=5, sigma=1.5)
                
                if outpaint_mask.sum() > 0:
                    ssim_loss = self.compute_ssim_loss(
                        pred_blur * outpaint_mask,
                        target_blur * outpaint_mask
                    )
                    losses['ssim'] = ssim_loss
                    total_loss += weights['ssim'] * ssim_loss
            
            if torch.rand(1).item() < 0.01:  
                components = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            
            return self.edge_weight * total_loss
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=device, requires_grad=True)