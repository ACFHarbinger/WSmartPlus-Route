# {py:mod}`src.policies.other.post_processing.ils`

```{py:module} src.policies.other.post_processing.ils
```

```{autodoc2-docstring} src.policies.other.post_processing.ils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IteratedLocalSearchPostProcessor <src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor
    :summary:
    ```
````

### API

`````{py:class} IteratedLocalSearchPostProcessor(ls_operator: typing.Union[str, typing.Dict[str, float]] = '2opt', perturbation_type: typing.Union[str, typing.Dict[str, float]] = 'double_bridge', n_restarts: int = 5, ls_iterations: int = 50, perturbation_strength: float = 0.2)
:canonical: src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor

Bases: {py:obj}`logic.src.interfaces.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.other.post_processing.ils.IteratedLocalSearchPostProcessor.process
```

````

`````
