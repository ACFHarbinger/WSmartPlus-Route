# {py:mod}`src.policies.other.post_processing.guided_local_search`

```{py:module} src.policies.other.post_processing.guided_local_search
```

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GuidedLocalSearchPostProcessor <src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor
    :summary:
    ```
````

### API

`````{py:class} GuidedLocalSearchPostProcessor(**kwargs: typing.Any)
:canonical: src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor

Bases: {py:obj}`logic.src.interfaces.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor.process
```

````

````{py:method} _get_operator_method(manager: typing.Any, name: str)
:canonical: src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor._get_operator_method

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor._get_operator_method
```

````

````{py:method} _update_penalties(routes: typing.List[typing.List[int]], dm: numpy.ndarray, penalty: numpy.ndarray)
:canonical: src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor._update_penalties

```{autodoc2-docstring} src.policies.other.post_processing.guided_local_search.GuidedLocalSearchPostProcessor._update_penalties
```

````

`````
