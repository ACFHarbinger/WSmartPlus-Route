# {py:mod}`src.pipeline.rl`

```{py:module} src.pipeline.rl
```

```{autodoc2-docstring} src.pipeline.rl
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.pipeline.rl.common
src.pipeline.rl.meta
src.pipeline.rl.hpo
src.pipeline.rl.core
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_rl_algorithm <src.pipeline.rl.get_rl_algorithm>`
  - ```{autodoc2-docstring} src.pipeline.rl.get_rl_algorithm
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RL_ALGORITHM_REGISTRY <src.pipeline.rl.RL_ALGORITHM_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.RL_ALGORITHM_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.pipeline.rl.__all__>`
  - ```{autodoc2-docstring} src.pipeline.rl.__all__
    :summary:
    ```
````

### API

````{py:data} RL_ALGORITHM_REGISTRY
:canonical: src.pipeline.rl.RL_ALGORITHM_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.RL_ALGORITHM_REGISTRY
```

````

````{py:function} get_rl_algorithm(name: str) -> type
:canonical: src.pipeline.rl.get_rl_algorithm

```{autodoc2-docstring} src.pipeline.rl.get_rl_algorithm
```
````

````{py:data} __all__
:canonical: src.pipeline.rl.__all__
:value: >
   ['REINFORCE', 'PPO', 'A2C', 'SAPO', 'GSPO', 'DRGRPO', 'GDPO', 'HRLModule', 'MetaRLModule', 'POMO', '...

```{autodoc2-docstring} src.pipeline.rl.__all__
```

````
