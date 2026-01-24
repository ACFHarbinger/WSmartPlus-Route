# {py:mod}`src.models.policies.classical.random_local_search`

```{py:module} src.models.policies.classical.random_local_search
```

```{autodoc2-docstring} src.models.policies.classical.random_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RandomLocalSearchPolicy <src.models.policies.classical.random_local_search.RandomLocalSearchPolicy>`
  - ```{autodoc2-docstring} src.models.policies.classical.random_local_search.RandomLocalSearchPolicy
    :summary:
    ```
````

### API

`````{py:class} RandomLocalSearchPolicy(env_name: str, n_iterations: int = 100, op_probs: dict[str, float] | None = None, **kwargs)
:canonical: src.models.policies.classical.random_local_search.RandomLocalSearchPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.classical.random_local_search.RandomLocalSearchPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.random_local_search.RandomLocalSearchPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', **kwargs) -> dict
:canonical: src.models.policies.classical.random_local_search.RandomLocalSearchPolicy.forward

```{autodoc2-docstring} src.models.policies.classical.random_local_search.RandomLocalSearchPolicy.forward
```

````

`````
