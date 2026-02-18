# {py:mod}`src.policies.other.post_processing.iterated_local_search`

```{py:module} src.policies.other.post_processing.iterated_local_search
```

```{autodoc2-docstring} src.policies.other.post_processing.iterated_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IteratedLocalSearchPostProcessor <src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor
    :summary:
    ```
````

### API

`````{py:class} IteratedLocalSearchPostProcessor(ls_operator: typing.Union[str, typing.Dict[str, float]] = '2opt', perturbation_type: typing.Union[str, typing.Dict[str, float]] = 'double_bridge', n_restarts: int = 5, ls_iterations: int = 50, perturbation_strength: float = 0.2)
:canonical: src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor

Bases: {py:obj}`logic.src.interfaces.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.other.post_processing.iterated_local_search.IteratedLocalSearchPostProcessor.process
```

````

`````
