# {py:mod}`src.policies.route_improvement.common.bandit`

```{py:module} src.policies.route_improvement.common.bandit
```

```{autodoc2-docstring} src.policies.route_improvement.common.bandit
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThompsonBandit <src.policies.route_improvement.common.bandit.ThompsonBandit>`
  - ```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit
    :summary:
    ```
````

### API

`````{py:class} ThompsonBandit(names: typing.List[str], seed: int = 42)
:canonical: src.policies.route_improvement.common.bandit.ThompsonBandit

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit.__init__
```

````{py:method} select() -> str
:canonical: src.policies.route_improvement.common.bandit.ThompsonBandit.select

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit.select
```

````

````{py:method} update(name: str, success: int)
:canonical: src.policies.route_improvement.common.bandit.ThompsonBandit.update

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit.update
```

````

````{py:method} get_weights() -> typing.Dict[str, typing.Tuple[float, float]]
:canonical: src.policies.route_improvement.common.bandit.ThompsonBandit.get_weights

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit.get_weights
```

````

````{py:method} load_weights(weights: typing.Dict[str, typing.Tuple[float, float]])
:canonical: src.policies.route_improvement.common.bandit.ThompsonBandit.load_weights

```{autodoc2-docstring} src.policies.route_improvement.common.bandit.ThompsonBandit.load_weights
```

````

`````
