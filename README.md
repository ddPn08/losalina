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
>
> Saving in `safetensors` format is not available.

Commandline arguments
```sh
python sd-scripts/train_network.py \
  --network_module losalina.hypernetwork \
  --network_args "layer_structure=[1, 2, 2, 1]" \
        "weight_init=KaimingNormal" \
        "activation_func=mish" \
        "add_layer_norm=True" \
        "dropout_structure=[0, 0.3, 0.3, 0]" \
        "last_layer_dropout=False"
```

Config file
```toml
network_module = "losalina.hypernetwork"
network_args = [
    "layer_structure=[1, 2, 2, 1]",
    "weight_init=KaimingNormal",
    "activation_func=mish",
    "add_layer_norm=True",
    "dropout_structure=[0, 0.3, 0.3, 0]",
    "last_layer_dropout=False",
]
```

||||
|-|-|-|
|`layer_structure`| Layer structure of hypernetwork.|ex.) `[1, 2, 2, 1]`|
|`weight_init`| Weight initialization method. | `Normal`, `XavierUniform`, `XavierNormal`, `KaimingUniform`, `KaimingNormal` |
|`activation_func`| Activation function. | `linear`, `relu`, `leakyrelu`, `elu`, `swish`, `tahh`, `sigmoid` ... |
|`add_layer_norm`| Add layer normalization. | `True`, `False` |
|`dropout_structure`| Dropout structure. | ex.) `[0, 0.3, 0.3, 0]` |
|`last_layer_dropout`| Add dropout to last layer. | `True`, `False` |
