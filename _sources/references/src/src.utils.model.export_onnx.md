# {py:mod}`src.utils.model.export_onnx`

```{py:module} src.utils.model.export_onnx
```

```{autodoc2-docstring} src.utils.model.export_onnx
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SingleInputWrapper <src.utils.model.export_onnx._SingleInputWrapper>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx._SingleInputWrapper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`export_encoder_to_onnx <src.utils.model.export_onnx.export_encoder_to_onnx>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx.export_encoder_to_onnx
    :summary:
    ```
* - {py:obj}`export_policy_components <src.utils.model.export_onnx.export_policy_components>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx.export_policy_components
    :summary:
    ```
* - {py:obj}`_infer_device <src.utils.model.export_onnx._infer_device>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx._infer_device
    :summary:
    ```
* - {py:obj}`_validate_onnx <src.utils.model.export_onnx._validate_onnx>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx._validate_onnx
    :summary:
    ```
* - {py:obj}`_try_simplify <src.utils.model.export_onnx._try_simplify>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx._try_simplify
    :summary:
    ```
* - {py:obj}`_build_arg_parser <src.utils.model.export_onnx._build_arg_parser>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx._build_arg_parser
    :summary:
    ```
* - {py:obj}`main <src.utils.model.export_onnx.main>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <src.utils.model.export_onnx.log>`
  - ```{autodoc2-docstring} src.utils.model.export_onnx.log
    :summary:
    ```
````

### API

````{py:data} log
:canonical: src.utils.model.export_onnx.log
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.utils.model.export_onnx.log
```

````

````{py:function} export_encoder_to_onnx(encoder: torch.nn.Module, export_dir: str = 'assets/onnx', filename: typing.Optional[str] = None, n_nodes: int = 50, embed_dim: int = 128, batch_size: int = 1, opset_version: int = 17, simplify: bool = False, verbose: bool = False, generator: typing.Optional[torch.Generator] = None) -> str
:canonical: src.utils.model.export_onnx.export_encoder_to_onnx

```{autodoc2-docstring} src.utils.model.export_onnx.export_encoder_to_onnx
```
````

````{py:function} export_policy_components(policy: torch.nn.Module, export_dir: str = 'assets/onnx', n_nodes: int = 50, embed_dim: int = 128, batch_size: int = 1, opset_version: int = 17, simplify: bool = False, component_names: typing.Optional[typing.List[str]] = None) -> typing.Dict[str, str]
:canonical: src.utils.model.export_onnx.export_policy_components

```{autodoc2-docstring} src.utils.model.export_onnx.export_policy_components
```
````

`````{py:class} _SingleInputWrapper(encoder: torch.nn.Module)
:canonical: src.utils.model.export_onnx._SingleInputWrapper

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.utils.model.export_onnx._SingleInputWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.model.export_onnx._SingleInputWrapper.__init__
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: src.utils.model.export_onnx._SingleInputWrapper.forward

```{autodoc2-docstring} src.utils.model.export_onnx._SingleInputWrapper.forward
```

````

`````

````{py:function} _infer_device(module: torch.nn.Module) -> torch.device
:canonical: src.utils.model.export_onnx._infer_device

```{autodoc2-docstring} src.utils.model.export_onnx._infer_device
```
````

````{py:function} _validate_onnx(path: str) -> None
:canonical: src.utils.model.export_onnx._validate_onnx

```{autodoc2-docstring} src.utils.model.export_onnx._validate_onnx
```
````

````{py:function} _try_simplify(path: str) -> None
:canonical: src.utils.model.export_onnx._try_simplify

```{autodoc2-docstring} src.utils.model.export_onnx._try_simplify
```
````

````{py:function} _build_arg_parser() -> argparse.ArgumentParser
:canonical: src.utils.model.export_onnx._build_arg_parser

```{autodoc2-docstring} src.utils.model.export_onnx._build_arg_parser
```
````

````{py:function} main() -> None
:canonical: src.utils.model.export_onnx.main

```{autodoc2-docstring} src.utils.model.export_onnx.main
```
````
