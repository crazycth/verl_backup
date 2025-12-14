QWEN4BLAYERS=[
    "model.embed_tokens.weight",
    "model.layers.35.self_attn.q_proj.weight",
    "model.layers.35.self_attn.k_proj.weight",
    "model.layers.35.self_attn.v_proj.weight",
    "model.layers.35.self_attn.o_proj.weight",
    "model.layers.35.self_attn.q_norm.weight",
    "model.layers.35.self_attn.k_norm.weight",
    "model.layers.35.mlp.gate_proj.weight",
    "model.layers.35.mlp.up_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
    "model.layers.35.input_layernorm.weight",
    "model.layers.35.post_attention_layernorm.weight"
]


QWEN1BLAYERS=[
    "model.embed_tokens.weight",
    "model.layers.27.self_attn.q_proj.weight",
    "model.layers.27.self_attn.k_proj.weight",
    "model.layers.27.self_attn.v_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.layers.27.self_attn.q_norm.weight",
    "model.layers.27.self_attn.k_norm.weight",
    "model.layers.27.mlp.gate_proj.weight",
    "model.layers.27.mlp.up_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.layers.27.input_layernorm.weight",
    "model.layers.27.post_attention_layernorm.weight"
]


QWEN8BLAYERS = [
    "lm_head.weight",
    "model.layers.35.self_attn.q_proj.weight",
    "model.layers.35.self_attn.k_proj.weight",
    "model.layers.35.self_attn.v_proj.weight",
    "model.layers.35.self_attn.o_proj.weight",
    "model.layers.35.self_attn.q_norm.weight",
    "model.layers.35.self_attn.k_norm.weight",
    "model.layers.35.mlp.gate_proj.weight",
    "model.layers.35.mlp.up_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
    "model.layers.35.input_layernorm.weight",
    "model.layers.35.post_attention_layernorm.weight"
]


NAME2LAYER = {
    "1B": QWEN1BLAYERS,
    "4B": QWEN4BLAYERS,
    "8B": QWEN8BLAYERS
}