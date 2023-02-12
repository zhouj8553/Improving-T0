import os
import sys
import json
import torch
from collections import OrderedDict


def inverse_transform_ckpt(model_directory, model_name_or_path):

    save_file_path = os.path.join(model_directory, "pytorch_model.bin")
    ################################## for test ###################################
    # if os.path.exists(os.path.join(model_directory,"back_converted_pytorch"))==False:
    #     os.makedirs(os.path.join(model_directory,"back_converted_pytorch"))
    # save_file_path = os.path.join(model_directory,"back_converted_pytorch", "pytorch_model.bin")
    ################################## for test ###################################
    if os.path.exists(save_file_path):
        print("\n\n")
        print("No need to transform ckpt!")
        print("\n\n")
        return

    prev_state_dict = \
        torch.load(os.path.join(model_directory, "mp_rank_00_model_states.pt"), map_location='cpu')["module"]
    ds_shape = {k: v.shape for (k, v) in prev_state_dict.items()}

    # original_model = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
    # original_model_shape = {k: v.shape for (k, v) in original_model.items()}

    original_config = json.load(open(os.path.join(model_name_or_path, "config.json")))
    num_layers = original_config["num_layers"]
    if "feed_forward_proj" in original_config and original_config["feed_forward_proj"] == "gated-gelu":
        use_gated_mlp = True
    else:
        use_gated_mlp = False


    state_dict = OrderedDict()

    encoder_word_embedding = prev_state_dict.pop("encoder.transformer.word_embeddings.weight")
    decoder_word_embedding = prev_state_dict.pop("decoder.transformer.word_embeddings.weight")
    assert encoder_word_embedding.equal(decoder_word_embedding)
    # assert original_model_shape["shared.weight"] == ds_shape["encoder.transformer.word_embeddings.weight"]
    state_dict["shared.weight"] = encoder_word_embedding

    state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = \
        prev_state_dict.pop("encoder.mixins.t5-attention.relative_attention_bias.weight")
    state_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = \
        prev_state_dict.pop("decoder.mixins.t5-attention.relative_attention_bias.weight")
    # assert original_model_shape["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] == \
    #       ds_shape["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]

    tie_word_embeddings = original_config["tie_word_embeddings"]
    if not tie_word_embeddings:
        state_dict[f"lm_head.weight"] = prev_state_dict.pop("decoder.mixins.t5-final.lm_head.weight")
        # assert original_model_shape[f"lm_head.weight"] == ds_shape[f"lm_head.weight"]

    for module in ["encoder", "decoder"]:
        state_dict[f"{module}.final_layer_norm.weight"] = \
            prev_state_dict.pop(f"{module}.transformer.final_layernorm.weight")
        # assert original_model_shape[f"{module}.final_layer_norm.weight"]==ds_shape[f"{
        # module}.final_layer_norm.weight"]

        for i in range(num_layers):
            state_dict[f"{module}.block.{i}.layer.0.layer_norm.weight"] = \
                prev_state_dict.pop(f"{module}.transformer.layers.{i}.input_layernorm.weight")
            # assert original_model_shape[f"{module}.block.{i}.layer.0.layer_norm.weight"] ==\
            #       ds_shape[f"{module}.block.{i}.layer.0.layer_norm.weight"]

            state_dict[f"{module}.block.{i}.layer.0.SelfAttention.o.weight"] = \
                prev_state_dict.pop(f"{module}.transformer.layers.{i}.attention.dense.weight")
            # assert original_model_shape[f"{module}.block.{i}.layer.0.SelfAttention.o.weight"]==\
            #        ds_shape[f"{module}.block.{i}.layer.0.SelfAttention.o.weight"]

            q_k_v = prev_state_dict.pop(f"{module}.transformer.layers.{i}.attention.query_key_value.weight")
            embedding_size = int(q_k_v.shape[0] / 3)
            query, key, value = torch.split(q_k_v, [embedding_size, embedding_size, embedding_size], dim=0)
            state_dict[f"{module}.block.{i}.layer.0.SelfAttention.q.weight"] = query
            state_dict[f"{module}.block.{i}.layer.0.SelfAttention.k.weight"] = key
            state_dict[f"{module}.block.{i}.layer.0.SelfAttention.v.weight"] = value

            # assert original_model_shape[f"{module}.block.{i}.layer.0.SelfAttention.q.weight"] == \
            #        ds_shape[f"{module}.block.{i}.layer.0.SelfAttention.q.weight"]
            # assert original_model_shape[f"{module}.block.{i}.layer.0.SelfAttention.k.weight"] == \
            #        ds_shape[f"{module}.block.{i}.layer.0.SelfAttention.k.weight"]
            # assert original_model_shape[f"{module}.block.{i}.layer.0.SelfAttention.v.weight"] == \
            #        ds_shape[f"{module}.block.{i}.layer.0.SelfAttention.v.weight"]

            state_dict[f"{module}.block.{i}.layer.1.layer_norm.weight"] = \
                prev_state_dict.pop(f"{module}.transformer.layers.{i}.post_attention_layernorm.weight")

            # assert original_model_shape[f"{module}.block.{i}.layer.1.layer_norm.weight"]==\
            #       ds_shape[f"{module}.block.{i}.layer.1.layer_norm.weight"]

            if module == "encoder":
                mlp_num = 1
            else:
                state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = \
                    prev_state_dict.pop(f"decoder.transformer.layers.{i}.cross_attention.query.weight")

                # assert original_model_shape[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] == \
                #        ds_shape[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"]

                key_value = prev_state_dict.pop(f"decoder.transformer.layers.{i}.cross_attention.key_value.weight")
                size = int(key_value.shape[0] / 2)
                k, v= torch.split(key_value, [size, size])
                state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k
                state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v

                # assert original_model_shape[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] == \
                #        ds_shape[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"]
                # assert original_model_shape[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] == \
                #        ds_shape[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"]

                state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = \
                    prev_state_dict.pop(f"decoder.transformer.layers.{i}.cross_attention.dense.weight")
                state_dict[f"decoder.block.{i}.layer.2.layer_norm.weight"] = \
                    prev_state_dict.pop(f"decoder.transformer.layers.{i}.post_cross_attention_layernorm.weight")
                mlp_num = 2

                # assert original_model_shape[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] == \
                #        ds_shape[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"]
                # assert original_model_shape[f"decoder.block.{i}.layer.2.layer_norm.weight"] == \
                #        ds_shape[f"decoder.block.{i}.layer.2.layer_norm.weight"]

            if use_gated_mlp:
                state_dict[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_0.weight"] = \
                    prev_state_dict.pop(f"{module}.transformer.layers.{i}.mlp.dense_h_to_4h.weight")
                state_dict[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_1.weight"] = \
                    prev_state_dict.pop(f"{module}.mixins.gated-mlp.gated_h_to_4h_list.{i}.weight")

                # assert original_model_shape[f"{module}.transformer.layers.{i}.mlp.dense_h_to_4h.weight"] == \
                #        ds_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_0.weight"]
                # assert original_model_shape[f"{module}.mixins.gated-mlp.gated_h_to_4h_list.{i}.weight"] == \
                #        ds_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi_1.weight"]

            else:
                state_dict[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi.weight"] = \
                    prev_state_dict.pop(f"{module}.transformer.layers.{i}.mlp.dense_h_to_4h.weight")

                # assert original_model_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi.weight"] == \
                #        ds_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wi.weight"]

            state_dict[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wo.weight"] = \
                prev_state_dict.pop(f"{module}.transformer.layers.{i}.mlp.dense_4h_to_h.weight")

            # assert original_model_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wo.weight"] == \
            #        ds_shape[f"{module}.block.{i}.layer.{mlp_num}.DenseReluDense.wo.weight"]

    print(prev_state_dict.keys())
    torch.save(state_dict, save_file_path)
    print("Finished transforming ckpt!")