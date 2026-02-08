# {py:mod}`src.interfaces.post_processing`

```{py:module} src.interfaces.post_processing
```

```{autodoc2-docstring} src.interfaces.post_processing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPostProcessor <src.interfaces.post_processing.IPostProcessor>`
  - ```{autodoc2-docstring} src.interfaces.post_processing.IPostProcessor
    :summary:
    ```
````

### API

`````{py:class} IPostProcessor
:canonical: src.interfaces.post_processing.IPostProcessor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.post_processing.IPostProcessor
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.interfaces.post_processing.IPostProcessor.process
:abstractmethod:

```{autodoc2-docstring} src.interfaces.post_processing.IPostProcessor.process
```

````

`````
