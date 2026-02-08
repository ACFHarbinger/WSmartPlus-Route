# {py:mod}`src.utils.functions.tensors`

```{py:module} src.utils.functions.tensors
```

```{autodoc2-docstring} src.utils.functions.tensors
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_to <src.utils.functions.tensors.move_to>`
  - ```{autodoc2-docstring} src.utils.functions.tensors.move_to
    :summary:
    ```
* - {py:obj}`compute_in_batches <src.utils.functions.tensors.compute_in_batches>`
  - ```{autodoc2-docstring} src.utils.functions.tensors.compute_in_batches
    :summary:
    ```
* - {py:obj}`do_batch_rep <src.utils.functions.tensors.do_batch_rep>`
  - ```{autodoc2-docstring} src.utils.functions.tensors.do_batch_rep
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <src.utils.functions.tensors.T>`
  - ```{autodoc2-docstring} src.utils.functions.tensors.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: src.utils.functions.tensors.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} src.utils.functions.tensors.T
```

````

````{py:function} move_to(var: src.utils.functions.tensors.T, device: torch.device, non_blocking: bool = False) -> src.utils.functions.tensors.T
:canonical: src.utils.functions.tensors.move_to

```{autodoc2-docstring} src.utils.functions.tensors.move_to
```
````

````{py:function} compute_in_batches(f: typing.Callable[..., typing.Any], calc_batch_size: int, *args: torch.Tensor, n: typing.Optional[int] = None) -> typing.Any
:canonical: src.utils.functions.tensors.compute_in_batches

```{autodoc2-docstring} src.utils.functions.tensors.compute_in_batches
```
````

````{py:function} do_batch_rep(v: typing.Any, n: int) -> typing.Any
:canonical: src.utils.functions.tensors.do_batch_rep

```{autodoc2-docstring} src.utils.functions.tensors.do_batch_rep
```
````
