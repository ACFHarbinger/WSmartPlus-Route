# {py:mod}`src.utils.ops.pomo`

```{py:module} src.utils.ops.pomo
```

```{autodoc2-docstring} src.utils.ops.pomo
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`select_start_nodes <src.utils.ops.pomo.select_start_nodes>`
  - ```{autodoc2-docstring} src.utils.ops.pomo.select_start_nodes
    :summary:
    ```
* - {py:obj}`select_start_nodes_by_distance <src.utils.ops.pomo.select_start_nodes_by_distance>`
  - ```{autodoc2-docstring} src.utils.ops.pomo.select_start_nodes_by_distance
    :summary:
    ```
* - {py:obj}`get_num_starts <src.utils.ops.pomo.get_num_starts>`
  - ```{autodoc2-docstring} src.utils.ops.pomo.get_num_starts
    :summary:
    ```
* - {py:obj}`get_best_actions <src.utils.ops.pomo.get_best_actions>`
  - ```{autodoc2-docstring} src.utils.ops.pomo.get_best_actions
    :summary:
    ```
````

### API

````{py:function} select_start_nodes(td: tensordict.TensorDict, num_starts: int) -> torch.Tensor
:canonical: src.utils.ops.pomo.select_start_nodes

```{autodoc2-docstring} src.utils.ops.pomo.select_start_nodes
```
````

````{py:function} select_start_nodes_by_distance(td: tensordict.TensorDict, num_starts: int) -> torch.Tensor
:canonical: src.utils.ops.pomo.select_start_nodes_by_distance

```{autodoc2-docstring} src.utils.ops.pomo.select_start_nodes_by_distance
```
````

````{py:function} get_num_starts(td: tensordict.TensorDict, env_name: typing.Optional[str] = None) -> int
:canonical: src.utils.ops.pomo.get_num_starts

```{autodoc2-docstring} src.utils.ops.pomo.get_num_starts
```
````

````{py:function} get_best_actions(actions: torch.Tensor, max_idxs: torch.Tensor) -> torch.Tensor
:canonical: src.utils.ops.pomo.get_best_actions

```{autodoc2-docstring} src.utils.ops.pomo.get_best_actions
```
````
