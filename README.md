# Losalina | Hypernetwork for kohya's sd-scripts

Train hypernetwork implemented in [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with [kohya_ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).

# Installation

## from GitHub

```
git clone https://github.com/ddPn08/losalina.git
cd losalina
pip install .
```

# How to use

> **Warning**
> Saving in `safetensors` format is not available.

```toml
network_module = losalina.hypernetwork
network_args = [
    "layer_structure=[1, 2, 2, 1]",
    "weight_init=KaimingNormal",
    "activation_func=mish",
    "add_layer_norm=True",
    "dropout_structure=[0, 0.3, 0.3, 0]",
    "last_layer_dropout=False",
]
```
