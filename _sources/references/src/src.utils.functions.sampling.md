# {py:mod}`src.utils.functions.sampling`

```{py:module} src.utils.functions.sampling
```

```{autodoc2-docstring} src.utils.functions.sampling
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sample_many <src.utils.functions.sampling.sample_many>`
  - ```{autodoc2-docstring} src.utils.functions.sampling.sample_many
    :summary:
    ```
````

### API

````{py:function} sample_many(inner_func: typing.Callable[..., typing.Any], get_cost_func: typing.Callable[..., typing.Any], input: typing.Any, batch_rep: int = 1, iter_rep: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.sampling.sample_many

```{autodoc2-docstring} src.utils.functions.sampling.sample_many
```
````
