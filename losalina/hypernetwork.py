# Hypernetwork module
#
# References:
# - https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/hypernetworks/hypernetwork.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/finetune/hypernetwork_nai.py


import os
from typing import *
import ast
import inspect
from typing import Any
import torch
from torch.nn.init import (
    normal_,
    xavier_uniform_,
    zeros_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
)


def create_network(multiplier, *args, **kwargs):
    enable_sizes = ast.literal_eval(
        kwargs.get("enable_sizes", "[320, 640, 768, 1024, 1280]")
    )
    layer_structure = ast.literal_eval(kwargs.get("layer_structure", "[1, 2, 1]"))
    activation_func = kwargs.get("activation_func", "linear")
    weight_init = kwargs.get("weight_init", "Normal")
    add_layer_norm = kwargs.get("add_layer_norm", False)
    dropout_structure = ast.literal_eval(kwargs.get("dropout_structure", "[]"))
    last_layer_dropout = kwargs.get("last_layer_dropout", False)

    if len(dropout_structure) < 1:
        dropout_structure = None

    print("Hyperenetwork options:")
    print(f"    multiplier: {multiplier}")
    print(f"    enable_sizes: {enable_sizes}")
    print(f"    layer_structure: {layer_structure}")
    print(f"    activation_func: {activation_func}")
    print(f"    weight_init: {weight_init}")
    print(f"    add_layer_norm: {add_layer_norm}")
    print(f"    dropout_structure: {dropout_structure}")
    print(f"    last_layer_dropout: {last_layer_dropout}")

    assert (
        len(layer_structure) > 1
    ), "hypernetwork layer_structure must be at least 2 layers"
    assert layer_structure[0] == 1, "hypernetwork layer_structure must start with 1"
    assert layer_structure[-1] == 1, "hypernetwork layer_structure must end with 1"
    assert dropout_structure is None or len(dropout_structure) == len(
        layer_structure
    ), "dropout_structure must be same length as layer_structure"

    return Hypernetwork(
        multiplier=multiplier,
        enable_sizes=enable_sizes,
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        dropout_structure=dropout_structure,
        last_layer_dropout=last_layer_dropout,
    )


def create_network_from_weights(
    multiplier, file, vae, text_encoder, unet, weights_sd=None, *args, **kwargs
):
    if weights_sd is None:
        weights_sd = torch.load(file, map_location="cpu")
    hypernetwork = Hypernetwork(multiplier)
    hypernetwork.restore_layers()
    hypernetwork.load_metadata(weights_sd)
    hypernetwork.load_layers()

    new_state_dict = {}
    for size, sd in weights_sd.items():
        if type(size) != int:
            continue
        for key, weights in enumerate(sd):
            prefix = f"{size}_{key}."
            for k, v in weights.items():
                new_state_dict[prefix + k] = v

    return hypernetwork, new_state_dict


activation_dict = {
    "linear": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "swish": torch.nn.Hardswish,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}
activation_dict.update(
    {
        cls_name.lower(): cls_obj
        for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation)
        if inspect.isclass(cls_obj)
        and cls_obj.__module__ == "torch.nn.modules.activation"
    }
)


def init_weight(
    layer: torch.nn.Linear,
    weight_init: str = "Normal",
    normal_std: int = 0.01,
    activation_func: str = None,
):
    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
        normal_(layer.weight.data, mean=0.0, std=normal_std)
        normal_(layer.bias.data, mean=0.0, std=0)
    elif weight_init == "XavierUniform":
        xavier_uniform_(layer.weight.data)
        zeros_(layer.bias.data)
    elif weight_init == "XavierNormal":
        xavier_normal_(layer.weight.data)
        zeros_(layer.bias.data)
    elif weight_init == "KaimingUniform":
        kaiming_uniform_(
            layer.weight.data,
            nonlinearity="leaky_relu" if "leakyrelu" == activation_func else "relu",
        )
        zeros_(layer.bias.data)
    elif weight_init == "KaimingNormal":
        kaiming_normal_(
            layer.weight.data,
            nonlinearity="leaky_relu" if "leakyrelu" == activation_func else "relu",
        )
        zeros_(layer.bias.data)
    else:
        raise KeyError(f"Key {weight_init} is not defined as initialization!")


def init_dropout(layer_structure: List[int], last_layer_dropout: bool):
    dropout_structure = [0] * len(layer_structure)
    if last_layer_dropout:
        dropout_structure[-1] = 0.3
    return dropout_structure


class HypernetworkModule(torch.nn.Module):
    def __init__(
        self,
        dim,
        multiplier: int = 1.0,
        layer_structure: List[int] = [1, 2, 1],
        activation_func: str = "linear",
        weight_init: str = "Normal",
        add_layer_norm: bool = False,
        dropout_structure: List[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.multiplier = multiplier

        layers = []

        for i in range(len(layer_structure) - 1):
            linear = torch.nn.Linear(
                dim * layer_structure[i], dim * layer_structure[i + 1]
            )
            init_weight(linear, weight_init, 0.01, activation_func)
            layers.append(linear)

            if (
                activation_func is not None
                and activation_func != "linear"
                and i < len(layer_structure) - 2
            ):
                layers.append(activation_dict[activation_func]())

            if add_layer_norm:
                norm = torch.nn.LayerNorm(dim * layer_structure[i + 1])
                init_weight(norm, None, 0.01)
                layers.append(norm)

            if dropout_structure is not None and dropout_structure[i + 1] > 0:
                assert (
                    0 < dropout_structure[i + 1] < 1
                ), "Dropout probability should be 0 or float between 0 and 1!"
                layers.append(torch.nn.Dropout(p=dropout_structure[i + 1]))

        self.linear = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.linear(x) * self.multiplier


class Hypernetwork(torch.nn.Module):
    def __init__(
        self,
        multiplier: int = 1.0,
        enable_sizes: List[int] = [320, 640, 768, 1024, 1280],
        layer_structure: List[int] = [1, 2, 1],
        activation_func: str = "linear",
        weight_init: str = "Normal",
        add_layer_norm: bool = False,
        dropout_structure: List[float] = None,
        last_layer_dropout: bool = False,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.enable_sizes = enable_sizes
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.dropout_structure = dropout_structure
        self.last_layer_dropout = last_layer_dropout

        if self.dropout_structure is None:
            self.dropout_structure = init_dropout(
                self.layer_structure, self.last_layer_dropout
            )

        self.layers: Dict[int, Tuple[torch.nn.Module, torch.nn.Module]] = {}
        self.load_layers()

    def load_layers(self):
        for size in self.enable_sizes:
            self.layers[size] = (
                HypernetworkModule(
                    size,
                    self.multiplier,
                    self.layer_structure,
                    self.activation_func,
                    self.weight_init,
                    self.add_layer_norm,
                    self.dropout_structure,
                ),
                HypernetworkModule(
                    size,
                    self.multiplier,
                    self.layer_structure,
                    self.activation_func,
                    self.weight_init,
                    self.add_layer_norm,
                    self.dropout_structure,
                ),
            )
            self.register_module(f"{size}_0", self.layers[size][0])
            self.register_module(f"{size}_1", self.layers[size][1])

    def restore_layers(self):
        self.layers = {}
        self._modules = {}

    def get_state_dict(self):
        state_dict = {}
        for _, size in enumerate(self.enable_sizes):
            state_dict[size] = [
                self.layers[size][0].state_dict(),
                self.layers[size][1].state_dict(),
            ]
        return state_dict

    def write_metadata(self, state_dict: Dict[str, Any]):
        state_dict["enable_sizes"] = self.enable_sizes
        state_dict["layer_structure"] = self.layer_structure
        state_dict["activation_func"] = self.activation_func
        state_dict["is_layer_norm"] = self.add_layer_norm
        state_dict["weight_initialization"] = self.weight_init
        state_dict["use_dropout"] = self.dropout_structure is not None
        state_dict["dropout_structure"] = self.dropout_structure
        state_dict["activate_output"] = False
        return state_dict

    def load_metadata(self, state_dict: Dict[str, Any]):
        self.enable_sizes = state_dict.get("enable_sizes", [320, 640, 768, 1024, 1280])
        self.layer_structure = state_dict.get("layer_structure", [1, 2, 1])
        self.activation_func = state_dict.get("activation_func", "linear")
        self.add_layer_norm = state_dict.get("is_layer_norm", False)
        self.weight_init = state_dict.get("weight_initialization", "Normal")
        self.dropout_structure = state_dict.get("dropout_structure", None)
        self.activate_output = state_dict.get("activate_output", False)

        if self.dropout_structure is None:
            self.dropout_structure = init_dropout(
                self.layer_structure, self.last_layer_dropout
            )

        return self

    def forward(self, x, context):
        size = context.shape[-1]
        assert size in self.enable_sizes
        module = self.layers[size]
        return module[0].forward(context), module[1].forward(context)

    def load_weights(self, file: str):
        state_dict = torch.load(file)
        self.restore_layers()
        self.load_metadata(state_dict)
        self.load_layers()
        self.load_state_dict(state_dict)
        return True

    def set_multiplier(self, multiplier: int):
        self.multiplier = multiplier
        for _, size in enumerate(self.enable_sizes):
            self.layers[size][0].multiplier = multiplier
            self.layers[size][1].multiplier = multiplier

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        blocks = unet.down_blocks + [unet.mid_block] + unet.up_blocks
        for block in blocks:
            if hasattr(block, "attentions"):
                for subblk in block.attentions:
                    if "SpatialTransformer" in str(
                        type(subblk)
                    ) or "Transformer2DModel" in str(
                        type(subblk)
                    ):  # 0.6.0 and 0.7~
                        for tf_block in subblk.transformer_blocks:
                            for attn in [tf_block.attn1, tf_block.attn2]:
                                size = attn.to_k.in_features
                                if size in self.enable_sizes:
                                    attn.hypernetwork = self
                                else:
                                    attn.hypernetwork = None
        setattr(unet, "__hypernetwork", True)

    def restore(self, *modules):
        for module in modules:
            if hasattr(module, "__hypernetwork"):
                delattr(module, "__hypernetwork")
            else:
                continue
            blocks = module.down_blocks + [module.mid_block] + module.up_blocks
            for block in blocks:
                if hasattr(block, "attentions"):
                    for subblk in block.attentions:
                        if "SpatialTransformer" in str(
                            type(subblk)
                        ) or "Transformer2DModel" in str(type(subblk)):
                            for tf_block in subblk.transformer_blocks:
                                for attn in [tf_block.attn1, tf_block.attn2]:
                                    if hasattr(attn, "hypernetwork"):
                                        delattr(attn, "hypernetwork")

    def prepare_optimizer_params(self, *args, **kwargs):
        return self.parameters()

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file: str, dtype: torch.dtype, metadata: Dict[str, Any]):
        if os.path.splitext(file)[1] == ".safetensors":
            raise ValueError("Hypernetwork weights should not be saved as safetensors.")
        state_dict = self.write_metadata(self.get_state_dict())
        if dtype is not None:
            for size in state_dict.keys():
                if size in self.enable_sizes:
                    for module in state_dict[size]:
                        for key in module.keys():
                            module[key] = (
                                module[key].detach().clone().to("cpu").to(dtype)
                            )
        torch.save(state_dict, file)
