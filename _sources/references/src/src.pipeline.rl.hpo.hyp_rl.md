# {py:mod}`src.pipeline.rl.hpo.hyp_rl`

```{py:module} src.pipeline.rl.hpo.hyp_rl
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HypRLHPO <src.pipeline.rl.hpo.hyp_rl.HypRLHPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO
    :summary:
    ```
````

### API

`````{py:class} HypRLHPO(cfg: logic.src.configs.Config, objective_fn: typing.Callable, search_space: typing.Optional[typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]] = None)
:canonical: src.pipeline.rl.hpo.hyp_rl.HypRLHPO

Bases: {py:obj}`src.pipeline.rl.hpo.base.BaseHPO`

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO.__init__
```

````{py:method} run() -> float
:canonical: src.pipeline.rl.hpo.hyp_rl.HypRLHPO.run

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO.run
```

````

````{py:method} _update_policy()
:canonical: src.pipeline.rl.hpo.hyp_rl.HypRLHPO._update_policy

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO._update_policy
```

````

````{py:method} _reset_episode()
:canonical: src.pipeline.rl.hpo.hyp_rl.HypRLHPO._reset_episode

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl.HypRLHPO._reset_episode
```

````

`````
