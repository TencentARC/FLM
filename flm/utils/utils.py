import torch
import torch.nn as nn
from flm.modules import heads, objectives, meter_utils


@torch.no_grad()
def adapt_vocab_size(state_dict, new_vocab_size):

    for name in state_dict.keys():
        if 'embeddings.word_embeddings.weight' in name or 'fusion_token_embedding.word_embeddings.weight' in name:
            expand_vocab(name, state_dict, new_vocab_size)

            # value = state_dict[name]
            # old_vocab_size, old_embed_dim = value.shape
            # if old_vocab_size != new_vocab_size:
            #     assert new_vocab_size > old_vocab_size
            #     new_embeddings = nn.Embedding(new_vocab_size, old_embed_dim)
            #     new_embeddings.apply(objectives.init_weights)
            #     new_embeddings.weight[:old_vocab_size] = value
            #     print(' replace vocab size of {} from {} to {}'.format(name ,old_vocab_size, new_vocab_size))
            #     state_dict[name] = new_embeddings.weight

        output_params = ['mlm_score', 'lm_score', 'lm_score_r', 'lm_score_f']

        for p in output_params:
            weight_name = p + '.decoder.weight'
            bias_name = p + '.bias'
            if weight_name in name or bias_name in name:
                expand_vocab(name, state_dict, new_vocab_size)

    return state_dict


def expand_vocab(name, state_dict, new_vocab_size):
    value = state_dict[name]
    if value.shape[0] != new_vocab_size:
        state_dict[name] = expand_tensor(value, new_vocab_size)
        print(' replace vocab size of {} from {} to {}'.format(
            name, value.shape[0], new_vocab_size))


def expand_tensor(value, new_vocab_size):
    if value.ndim == 1:
        old_vocab_size = value.shape[0]
        new_embeddings = torch.zeros(new_vocab_size)
    else:
        old_vocab_size, old_embed_dim = value.shape
        new_embeddings = torch.zeros(new_vocab_size, old_embed_dim)
    assert new_vocab_size > old_vocab_size

    new_embeddings.data.normal_(mean=0.0, std=0.02)

    new_embeddings[:old_vocab_size] = value
    return new_embeddings
