import peft
import torch
import safetensors.torch


def determine_adapter_target_modules(args, unet, transformer):
    if unet is not None:
        return ["to_k", "to_q", "to_v", "to_out.0"]
    elif transformer is not None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        if args.model_family.lower() == "flux" and args.flux_lora_target == "all":
            # target_modules = mmdit layers here
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context+ffs":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        elif args.flux_lora_target == "all+ffs":
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "ai-toolkit":
            # from ostris' ai-toolkit, possibly required to continue finetuning one.
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "norm.linear",
                "norm1.linear",
                "norm1_context.linear",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "tiny":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
                "single_transformer_blocks.20.proj_out",
            ]
        elif args.flux_lora_target == "nano":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
            ]

        # Add presets here
        # The following are a few preset examples
        # In order to activate these presets, a few lines of code are also needed in SimpleTuner/configure.py (line 528), and SimpleTuner/helpers/configuration/cmd_args.py (line 131)

        elif args.flux_lora_target == "preset1":
            target_modules = [
                "single_transformer_blocks.5.proj_out",
                "single_transformer_blocks.6.proj_out",
                "single_transformer_blocks.15.proj_out",
                "single_transformer_blocks.16.proj_out",
                "single_transformer_blocks.25.proj_out",
                "single_transformer_blocks.26.proj_out",
            ]

        elif args.flux_lora_target == "preset2":
            target_modules = [
                "single_transformer_blocks.8.proj_out",
                "single_transformer_blocks.9.proj_out",
                "single_transformer_blocks.18.proj_out",
                "single_transformer_blocks.19.proj_out",
                "single_transformer_blocks.28.proj_out",
                "single_transformer_blocks.29.proj_out",
            ]

        elif args.flux_lora_target == "preset3":
            target_modules = [
                "single_transformer_blocks.1.proj_out",
                "single_transformer_blocks.2.proj_out",
                "single_transformer_blocks.11.proj_out",
                "single_transformer_blocks.12.proj_out",
                "single_transformer_blocks.21.proj_out",
                "single_transformer_blocks.22.proj_out",
            ]

        elif args.flux_lora_target == "preset4":
            target_modules = [
                "single_transformer_blocks.3.proj_out",
                "single_transformer_blocks.4.proj_out",
                "single_transformer_blocks.13.proj_out",
                "single_transformer_blocks.14.proj_out",
                "single_transformer_blocks.23.proj_out",
                "single_transformer_blocks.24.proj_out",
            ]

        elif args.flux_lora_target == "faces":
            target_modules = [
                "single_transformer_blocks.7.proj_out",
                "single_transformer_blocks.17.proj_out",
                "single_transformer_blocks.27.proj_out",
            ]

        elif args.flux_lora_target == "poses":
            target_modules = [
                "single_transformer_blocks.10.proj_out",
                "single_transformer_blocks.20.proj_out",
                "single_transformer_blocks.30.proj_out",
            ]

        elif args.flux_lora_target == "poses_large":
            target_modules = [
                "single_transformer_blocks.7.proj_out",
                "single_transformer_blocks.10.proj_out",
                "single_transformer_blocks.17.proj_out",
                "single_transformer_blocks.20.proj_out",
                "single_transformer_blocks.27.proj_out",
                "single_transformer_blocks.30.proj_out",
            ]

        return target_modules


@torch.no_grad()
def load_lora_weights(dictionary, filename, loraKey="default", use_dora=False):
    additional_keys = set()
    state_dict = safetensors.torch.load_file(filename)
    for prefix, model in dictionary.items():
        lora_layers = {
            (prefix + "." + x): y
            for (x, y) in model.named_modules()
            if isinstance(y, peft.tuners.lora.layer.Linear)
        }
    missing_keys = set(
        [x + ".lora_A.weight" for x in lora_layers.keys()]
        + [x + ".lora_B.weight" for x in lora_layers.keys()]
        + (
            [x + ".lora_magnitude_vector.weight" for x in lora_layers.keys()]
            if use_dora
            else []
        )
    )
    for k, v in state_dict.items():
        if "lora_A" in k:
            kk = k.replace(".lora_A.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_A[loraKey].weight.copy_(v)
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif "lora_B" in k:
            kk = k.replace(".lora_B.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_B[loraKey].weight.copy_(v)
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif ".alpha" in k or ".lora_alpha" in k:
            kk = k.replace(".lora_alpha", "").replace(".alpha", "")
            if kk in lora_layers:
                lora_layers[kk].lora_alpha[loraKey] = v
        elif ".lora_magnitude_vector" in k:
            kk = k.replace(".lora_magnitude_vector.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_magnitude_vector[loraKey].weight.copy_(v)
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
    return (additional_keys, missing_keys)
