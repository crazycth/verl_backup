import torch
import torch.nn.functional as F


def compute_phi(
    logits: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_v: int | None = None,
    chunk_t: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Compute pairwise phi for each response token.

    Args:
        logits: (B, L, V) pre-softmax logits over vocab for response positions.
        responses: (B, L) token ids for the responses.
        response_mask: (B, L) mask (1 for valid tokens, 0 for padding).
        chunk_v: optional vocab chunk size to reduce peak memory.
        chunk_t: optional time chunk size to reduce L^2 peak memory.
        dtype: dtype for softmax computation.

    Returns:
        phi: (B, L, L) tensor masked on padding.
    """
    if logits is None:
        raise ValueError("logits is required for phi computation")

    B, L, V = logits.shape
    logits = logits.to(dtype)
    responses = responses.to(logits.device)
    response_mask = response_mask.to(logits.device)

    # Compute probabilities
    if chunk_v is None:
        pi = F.softmax(logits, dim=-1)
    else:
        # Chunked softmax over vocab to reduce memory
        max_logits = logits.max(dim=-1, keepdim=True).values
        exp_parts = []
        exp_sum = None
        for start in range(0, V, chunk_v):
            end = min(start + chunk_v, V)
            logits_slice = logits[..., start:end]
            exp_slice = torch.exp(logits_slice - max_logits)
            exp_parts.append(exp_slice)
            exp_slice_sum = exp_slice.sum(dim=-1, keepdim=True)
            exp_sum = exp_slice_sum if exp_sum is None else exp_sum + exp_slice_sum
        pi_parts = [part / exp_sum for part in exp_parts]
        pi = torch.cat(pi_parts, dim=-1)

    # Gather pi_j(a_k); broadcast responses across j dimension
    responses_exp = responses.unsqueeze(1).expand(-1, L, -1)  # (B, L, L)
    pi_j_a_k = pi.gather(-1, responses_exp)  # (B, L, L)
    pi_k_a_j = pi_j_a_k.transpose(1, 2)  # symmetry

    # Indicator I[a_j == a_k]
    actions_eq = (responses.unsqueeze(1) == responses.unsqueeze(2)).to(pi.dtype)

    # <pi_j, pi_k> via vocab inner product
    if chunk_t is None:
        pi_inner = pi @ pi.transpose(-1, -2)  # (B, L, L)
    else:
        pi_inner = logits.new_zeros((B, L, L), dtype=pi.dtype)
        for t_start in range(0, L, chunk_t):
            t_end = min(t_start + chunk_t, L)
            pi_block = pi[:, t_start:t_end, :]  # (B, t, V)
            pi_inner[:, t_start:t_end, :] = pi_block @ pi.transpose(-1, -2)

    phi = actions_eq - pi_j_a_k - pi_k_a_j + pi_inner

    # Mask padding rows/cols
    mask_row = response_mask.unsqueeze(2)
    mask_col = response_mask.unsqueeze(1)
    valid_mask = mask_row * mask_col
    phi = phi * valid_mask

    return phi
