# {py:mod}`src.models.subnets.decoders.common.selection`

```{py:module} src.models.subnets.decoders.common.selection
```

```{autodoc2-docstring} src.models.subnets.decoders.common.selection
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`select_action <src.models.subnets.decoders.common.selection.select_action>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.common.selection.select_action
    :summary:
    ```
* - {py:obj}`select_action_log_prob <src.models.subnets.decoders.common.selection.select_action_log_prob>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.common.selection.select_action_log_prob
    :summary:
    ```
````

### API

````{py:function} select_action(probs: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, strategy: str = 'greedy') -> torch.Tensor
:canonical: src.models.subnets.decoders.common.selection.select_action

```{autodoc2-docstring} src.models.subnets.decoders.common.selection.select_action
```
````

````{py:function} select_action_log_prob(log_probs: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, strategy: str = 'greedy') -> torch.Tensor
:canonical: src.models.subnets.decoders.common.selection.select_action_log_prob

```{autodoc2-docstring} src.models.subnets.decoders.common.selection.select_action_log_prob
```
````
