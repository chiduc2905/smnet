"""Debug utilities for model verification.

Usage:
    from function.debug_utils import print_grad_norm, stat

In training loop after loss.backward():
    print_grad_norm(model, epoch, step)

In model.encode():
    stat(x, "after_stage_name")
"""
import torch


def print_grad_norm(model, epoch=0, step=0, print_every=10):
    """Print gradient norms for all parameters.
    
    Call after loss.backward() to check gradient flow.
    
    Expected values:
        - Conv / ChannelProj: ~1e-3 → 1e-2
        - Mamba params: ~1e-4 → 1e-3
        - Attention / gating: should NOT be 0
    
    Args:
        model: The model
        epoch: Current epoch (for filtering)
        step: Current step within epoch
        print_every: Only print every N steps
    """
    if step % print_every != 0:
        return
    
    print(f"\n[Epoch {epoch} Step {step}] Gradient Norms:")
    print("-" * 60)
    
    grad_stats = {}
    zero_grad_layers = []
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_mean = p.grad.abs().mean().item()
            grad_stats[name] = grad_mean
            if grad_mean < 1e-10:
                zero_grad_layers.append(name)
        else:
            grad_stats[name] = None
    
    # Group by layer type
    groups = {
        'conv': [],
        'mamba': [],
        'attn': [],
        'other': []
    }
    
    for name, grad in grad_stats.items():
        if grad is None:
            continue
        if 'conv' in name.lower() or 'proj' in name.lower():
            groups['conv'].append((name, grad))
        elif 'mamba' in name.lower() or 'ss2d' in name.lower():
            groups['mamba'].append((name, grad))
        elif 'attn' in name.lower() or 'gate' in name.lower():
            groups['attn'].append((name, grad))
        else:
            groups['other'].append((name, grad))
    
    for group_name, params in groups.items():
        if params:
            grads = [g for _, g in params]
            avg = sum(grads) / len(grads)
            print(f"  {group_name:10s}: avg={avg:.2e}, min={min(grads):.2e}, max={max(grads):.2e}")
    
    if zero_grad_layers:
        print(f"  ⚠️ ZERO GRAD LAYERS ({len(zero_grad_layers)}):")
        for name in zero_grad_layers[:5]:  # Show first 5
            print(f"      {name}")
    
    print("-" * 60)


def stat(x, name):
    """Print feature statistics (mean, std).
    
    Call this at each stage in model.encode() to check for feature collapse.
    
    Expected std values:
        - ConvBlocks: 0.8 – 1.5
        - PatchMerge: 0.6 – 1.2
        - DualBranch: 0.5 – 1.0
        - UnifiedAttn: should NOT be < 0.3
    
    Args:
        x: Feature tensor (B, C, H, W) or (B, N, C)
        name: Stage name for printing
    """
    mean = x.mean().item()
    std = x.std().item()
    min_val = x.min().item()
    max_val = x.max().item()
    
    # Warning if std is too low (feature collapse)
    warning = "⚠️ COLLAPSE!" if std < 0.3 else ""
    
    print(f"  [{name:20s}] mean={mean:+.4f}, std={std:.4f}, range=[{min_val:.3f}, {max_val:.3f}] {warning}")


def print_logit_stats(logits, step=0, print_every=10):
    """Print logit statistics before softmax.
    
    Expected:
        - range ~[-5, 5] initially
        - not all zero
        - not all same value
    
    Args:
        logits: (N, Way) logits before softmax
        step: Current step
        print_every: Print frequency
    """
    if step % print_every != 0:
        return
    
    min_val = logits.min().item()
    max_val = logits.max().item()
    mean_val = logits.mean().item()
    std_val = logits.std().item()
    
    warning = ""
    if std_val < 0.1:
        warning = "⚠️ LOGITS TOO SIMILAR - similarity not discriminating!"
    elif abs(max_val - min_val) < 0.1:
        warning = "⚠️ LOGITS FLAT - no class separation!"
    
    print(f"  [Logits] min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}, std={std_val:.4f} {warning}")


# Global debug flag
DEBUG_MODE = False


def set_debug_mode(enabled: bool):
    """Enable/disable debug printing."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    print(f"Debug mode: {'ON' if enabled else 'OFF'}")


def is_debug_mode():
    """Check if debug mode is enabled."""
    return DEBUG_MODE
