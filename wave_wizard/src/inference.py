import torch


def enhance(
    model: torch.nn.Module, noisy: torch.Tensor, sample_len=16384
) -> torch.Tensor:
    """efficient batch enchance

    Args:
        model (torch.nn.Module): denoising model
        noisy (torch.Tensor): one batch noisy tensor
        sample_len (int, optional): length of optimal batch. Defaults to 16384.

    Returns:
        torch.Tensor: _description_
    """
    if noisy.size(-1) % sample_len != 0:
        padded_length = sample_len - (noisy.size(-1) % sample_len)
        noisy = torch.cat([noisy, torch.zeros(size=(1, 1, padded_length))], dim=-1)

    assert noisy.size(-1) % sample_len == 0 and noisy.dim() == 3

    noisy_chunks = list(torch.split(noisy, sample_len, dim=-1))
    noisy_chunks = torch.cat(noisy_chunks, dim=0)

    enhanced_chunks = model(noisy_chunks).detach().cpu()

    enhanced = enhanced_chunks.reshape(-1)
    if padded_length != 0:
        enhanced = enhanced[:-padded_length]
        noisy = noisy[:-padded_length]

    return enhanced
