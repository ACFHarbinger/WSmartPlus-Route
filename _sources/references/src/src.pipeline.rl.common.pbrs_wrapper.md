# {py:mod}`src.pipeline.rl.common.pbrs_wrapper`

```{py:module} src.pipeline.rl.common.pbrs_wrapper
```

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PBRSShaper <src.pipeline.rl.common.pbrs_wrapper.PBRSShaper>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_potential_vrpp <src.pipeline.rl.common.pbrs_wrapper.get_potential_vrpp>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.get_potential_vrpp
    :summary:
    ```
* - {py:obj}`_get_potential_not_implemented <src.pipeline.rl.common.pbrs_wrapper._get_potential_not_implemented>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper._get_potential_not_implemented
    :summary:
    ```
* - {py:obj}`get_potential_fn <src.pipeline.rl.common.pbrs_wrapper.get_potential_fn>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.get_potential_fn
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.pbrs_wrapper.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.logger
    :summary:
    ```
* - {py:obj}`_POTENTIAL_REGISTRY <src.pipeline.rl.common.pbrs_wrapper._POTENTIAL_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper._POTENTIAL_REGISTRY
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.pbrs_wrapper.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.logger
```

````

````{py:function} get_potential_vrpp(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.pipeline.rl.common.pbrs_wrapper.get_potential_vrpp

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.get_potential_vrpp
```
````

````{py:function} _get_potential_not_implemented(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.pipeline.rl.common.pbrs_wrapper._get_potential_not_implemented

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper._get_potential_not_implemented
```
````

````{py:data} _POTENTIAL_REGISTRY
:canonical: src.pipeline.rl.common.pbrs_wrapper._POTENTIAL_REGISTRY
:type: dict[str, typing.Callable[[tensordict.TensorDict], torch.Tensor]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper._POTENTIAL_REGISTRY
```

````

````{py:function} get_potential_fn(env_name: str) -> typing.Callable[[tensordict.TensorDict], torch.Tensor]
:canonical: src.pipeline.rl.common.pbrs_wrapper.get_potential_fn

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.get_potential_fn
```
````

`````{py:class} PBRSShaper(gamma: float, env_name: str, shaping_weight: float = 1.0, potential_fn: typing.Optional[typing.Callable[[tensordict.TensorDict], torch.Tensor]] = None)
:canonical: src.pipeline.rl.common.pbrs_wrapper.PBRSShaper

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.__init__
```

````{py:method} record_initial(td: tensordict.TensorDict) -> None
:canonical: src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.record_initial

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.record_initial
```

````

````{py:method} apply(base_reward: torch.Tensor, final_td: tensordict.TensorDict) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.apply

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.apply
```

````

````{py:method} reset() -> None
:canonical: src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.reset

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.reset
```

````

````{py:method} __repr__() -> str
:canonical: src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.__repr__

```{autodoc2-docstring} src.pipeline.rl.common.pbrs_wrapper.PBRSShaper.__repr__
```

````

`````
