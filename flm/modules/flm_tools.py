import torch
import torch.nn.functional as F


def get_corr_bi_attention_mask(mask, mask_r, span_corr_rate=0):
    """prepare the attention mask in reconstrctor"""
    bs, L, M, N = mask.shape
    org_bi_mask = torch.cat([mask, mask_r], dim=-1)
    bi_mask = org_bi_mask.detach().clone()
    bi_mask[:, :, torch.arange(1, N), torch.arange(1, N)] = -10000.
    bi_mask[:, :, torch.arange(
        1, N), N + torch.arange(1, N)] = -10000.  # [bs, L, L]
    text_len = (bi_mask != -10000.).sum(dim=3) + 1
    text_len[:, :, 0] = 1

    if span_corr_rate > 0:
        add_corr_rate = torch.maximum(torch.zeros_like(
            text_len), (text_len * span_corr_rate - 1.)/(text_len - 1 + 1e-5))
        mask_num = torch.distributions.Binomial(
            text_len.float() - 1, add_corr_rate).sample().int()
        start_bias = mask_num // 2 + torch.bernoulli(mask_num/2 - mask_num//2)
        angle = torch.arange(0, N, device=mask.device).long()
        start = torch.maximum(angle - start_bias.long(), 0*angle)
        end = torch.minimum(start + N + mask_num, start.new_tensor(2*N-1))
        start_step = angle[None, None].repeat(bs, L, 1) - start
        for i in range(torch.max(start_step[:, :, 1:])):
            bi_mask[torch.arange(bs).reshape(bs, 1, 1).repeat(1, L, N), torch.arange(L).reshape(1, L, 1).repeat(
                bs, 1, N), angle[None, None].repeat(bs, L, 1), torch.minimum(start+i, angle[None, None])] = -10000.

        end_step = end - angle[None, None].repeat(bs, L, 1) - N
        for i in range(torch.max(end_step[:, :, 1:])):
            bi_mask[torch.arange(bs).reshape(bs, 1, 1).repeat(1, L, N), torch.arange(L).reshape(1, L, 1).repeat(
                bs, 1, N), angle[None, None].repeat(bs, L, 1), torch.maximum(end-i, N + angle[None, None])] = -10000.
    return torch.cat([org_bi_mask[:, :, :1], bi_mask[:, :, 1:]], dim=2)
