import random
import torch
from utils import print_rank_0
from SwissArmyTransformer.model import GLMModel, EncoderDecoderModel
from SwissArmyTransformer.model.mixins import BaseMixin
from SwissArmyTransformer.model.base_model import non_conflict
from SwissArmyTransformer.mpu.transformer import standard_attention, split_tensor_along_last_dim


def get_prefix_model(model_cls):
    def new_model_cls(args):
        model = model_cls(args)
        if args.freeze_transformer:
            print_rank_0("Freeze transformer parameters")
            model.requires_grad_(False)
        prefix_mixin = PrefixPromptMixin(prefix_length=args.prefix_prompt, num_layers=args.num_layers,
                                         hidden_size=args.hidden_size, prompt_func=args.prompt_func)
        if isinstance(model, GLMModel):
            model.add_mixin(
                "prefix_prompt",
                prefix_mixin
            )
        elif isinstance(model, EncoderDecoderModel):
            model.encoder.add_mixin(
                "prefix_prompt",
                prefix_mixin
            )
        else:
            raise NotImplementedError(model_cls)
        return model

    return new_model_cls


class PrefixPromptMixin(BaseMixin):
    def __init__(self, prefix_length, num_layers, hidden_size, prompt_func='none'):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(prefix_length, hidden_size) for _ in range(self.num_layers)]
        )
        for embedding in self.embeddings:
            torch.nn.init.normal_(embedding.weight, mean=0, std=0.1)
        self.prompt_func = prompt_func
        if prompt_func != 'none':
            raise NotImplementedError(prompt_func)

    def get_prompt(self, batch_size, layer_id):
        prefix_hidden_states = self.embeddings[layer_id].weight
        prefix_hidden_states = prefix_hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
        return prefix_hidden_states

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, old_impl=standard_attention, layer_id=None, cross_attention=False,
                     **kw_args):
        if not cross_attention:
            attn_module = self.transformer.layers[layer_id].attention
            prefix = self.get_prompt(q.size(0), layer_id)
            mixed_raw_layer = attn_module.query_key_value(prefix)
            (_,
             mixed_key_prefix,
             mixed_value_prefix) = split_tensor_along_last_dim(mixed_raw_layer, 3)
            key_prefix = attn_module._transpose_for_scores(mixed_key_prefix)
            value_prefix = attn_module._transpose_for_scores(mixed_value_prefix)
            k, v = torch.cat((key_prefix, k), dim=2), torch.cat((value_prefix, v), dim=2)
            if mask.numel() > 1:
                mask_prefix = torch.ones(self.prefix_length, device=mask.device, dtype=mask.dtype)
                mask_prefix = mask_prefix.expand(*(mask.size()[:-1]), -1)
                mask = torch.cat((mask_prefix, mask), dim=-1)
        return old_impl(q, k, v, mask, dropout_fn, layer_id=layer_id, cross_attention=cross_attention,
                        **kw_args)
