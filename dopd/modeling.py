import torch
import transformers
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
_a2d_model_registered = False
class A2DQwen3Config(transformers.Qwen3Config):
    model_type = 'a2d-qwen3'

def _register_a2d_model_classes():
    global _a2d_model_registered
    if _a2d_model_registered:
        return
    _a2d_model_registered = True

    class A2DQwen3Model(transformers.Qwen3Model):

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError()
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            if isinstance(attention_mask, dict):
                causal_mask_mapping = attention_mask
            else:
                if not (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4):
                    if attention_mask is None:
                        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
                    attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
                causal_mask_mapping = {'full_attention': attention_mask}
                if hasattr(self, 'has_sliding_layers') and self.has_sliding_layers:
                    causal_mask_mapping['sliding_attention'] = attention_mask
            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers[:self.config.num_hidden_layers]:
                layer_mask = causal_mask_mapping.get(getattr(layer, 'attention_type', 'full_attention'), attention_mask)
                out = layer(hidden_states, attention_mask=layer_mask, position_ids=position_ids, past_key_value=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
                hidden_states = out[0] if isinstance(out, tuple) else out
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

    class A2DQwen3LMHeadModel(transformers.Qwen3ForCausalLM):
        config_class = A2DQwen3Config

        def __init__(self, config):
            transformers.Qwen3PreTrainedModel.__init__(self, config)
            self.model = A2DQwen3Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()
    transformers.AutoModel.register(A2DQwen3Config, A2DQwen3LMHeadModel)
    transformers.AutoModelForMaskedLM.register(A2DQwen3Config, A2DQwen3LMHeadModel)
transformers.AutoConfig.register("a2d-qwen3", A2DQwen3Config)
