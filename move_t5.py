import os
import sys
import json
import torch
from collections import OrderedDict

# root_directory = "../../huggingface_models"
def convert_hf_to_ds(root_directory,model_name):
    model_directory = os.path.join(root_directory, model_name)
    save_dir = os.path.join(model_directory, "mp_rank_00_model_states.pt")
    if os.path.exists(save_dir)==True:
        return 
    hg_state_dict = torch.load(os.path.join(model_directory, "pytorch_model.bin"))
    config = json.load(open(os.path.join(model_directory, "config.json")))
    if "feed_forward_proj" in config and config["feed_forward_proj"] == "gated-gelu":
        use_gated_mlp = True
    else:
        use_gated_mlp = False
    num_layers = config["num_layers"]
    tie_word_embeddings = config["tie_word_embeddings"]
    state_dict = OrderedDict()

    word_embedding = hg_state_dict.pop("shared.weight")
    state_dict["encoder.transformer.word_embeddings.weight"] = word_embedding
    state_dict["decoder.transformer.word_embeddings.weight"] = word_embedding

    state_dict["encoder.mixins.t5-attention.relative_attention_bias.weight"] = \
        hg_state_dict.pop("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")

    state_dict["decoder.mixins.t5-attention.relative_attention_bias.weight"] = \
        hg_state_dict.pop("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")

    if not tie_word_embeddings:
        state_dict["decoder.mixins.t5-final.lm_head.weight"] = \
            hg_state_dict.pop(f"lm_head.weight")

    for module in ["encoder", "decoder"]:
        state_dict[f"{module}.transformer.final_layernorm.weight"] = \
            hg_state_dict.pop(f"{module}.final_layer_norm.weight")
        for i in range(num_layers):
            state_dict[f"{module}.transformer.layers.{i}.input_layernorm.weight"] = \
                hg_state_dict.pop(f"{module}.block.{i}.layer.0.layer_norm.weight")
            state_dict[f"{module}.transformer.layers.{i}.attention.dense.weight"] = \
                hg_state_dict.pop(f"{module}.block.{i}.layer.0.SelfAttention.o.weight")
            query, key, value = hg_state_dict.pop(f"{module}.block.{i}.layer.0.SelfAttention.q.weight"), \
                                hg_state_dict.pop(f"{module}.block.{i}.layer.0.SelfAttention.k.weight"), \
                                hg_state_dict.pop(f"{module}.block.{i}.layer.0.SelfAttention.v.weight")
            state_dict[f"{module}.transformer.layers.{i}.attention.query_key_value.weight"] = torch.cat((query, key, value),
                                                                                                        dim=0)
            state_dict[f"{module}.transformer.layers.{i}.post_attention_layernorm.weight"] = \
                hg_state_dict.pop(f"{module}.block.{i}.layer.1.layer_norm.weight")
            if module == "encoder":
                mlp_num = 1
            else:
                state_dict[f"decoder.transformer.layers.{i}.cross_attention.query.weight"] = \
                    hg_state_dict.pop(f"decoder.block.{i}.layer.1.EncDecAttention.q.weight")
                key, value = hg_state_dict.pop(f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"), \
                            hg_state_dict.pop(f"decoder.block.{i}.layer.1.EncDecAttention.v.weight")
                state_dict[f"decoder.transformer.layers.{i}.cross_attention.key_value.weight"] = torch.cat((key, value),
                                                                                                        dim=0)
                state_dict[f"decoder.transformer.layers.{i}.cross_attention.dense.weight"] = \
                    hg_state_dict.pop(f"decoder.block.{i}.layer.1.EncDecAttention.o.weight")
                state_dict[f"decoder.transformer.layers.{i}.post_cross_attention_layernorm.weight"] = \
                    hg_state_dict.pop(f"decoder.block.{i}.layer.2.layer_norm.weight")
                mlp_num = 2
            if use_gated_mlp:
                state_dict[f"{module}.transformer.layers.{i}.mlp.dense_h_to_4h.weight"] = \
                    hg_state_dict.pop(f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_0.weight")
                state_dict[f"{module}.mixins.gated-mlp.gated_h_to_4h_list.{i}.weight"] = \
                    hg_state_dict.pop(f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_1.weight")
            else:
                state_dict[f"{module}.transformer.layers.{i}.mlp.dense_h_to_4h.weight"] = \
                    hg_state_dict.pop(f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi.weight")
            state_dict[f"{module}.transformer.layers.{i}.mlp.dense_4h_to_h.weight"] = \
                hg_state_dict.pop(f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wo.weight")
    checkpoint = {"module": state_dict}
    print(hg_state_dict.keys())
    torch.save(checkpoint, os.path.join(model_directory, "mp_rank_00_model_states.pt"))

if __name__=='__main__':
    root_directory = sys.argv[1]
    model_name = sys.argv[2]
    # import pdb 
    # pdb.set_trace()
    convert_hf_to_ds(root_directory,model_name)
    
# python move_t5.py ../../../huggingface_models t5-large-lm-adapt
