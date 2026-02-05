# {py:mod}`src.policies.post_processing`

```{py:module} src.policies.post_processing
```

```{autodoc2-docstring} src.policies.post_processing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPostProcessor <src.policies.post_processing.IPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.IPostProcessor
    :summary:
    ```
* - {py:obj}`PostProcessorRegistry <src.policies.post_processing.PostProcessorRegistry>`
  - ```{autodoc2-docstring} src.policies.post_processing.PostProcessorRegistry
    :summary:
    ```
* - {py:obj}`PostProcessorFactory <src.policies.post_processing.PostProcessorFactory>`
  - ```{autodoc2-docstring} src.policies.post_processing.PostProcessorFactory
    :summary:
    ```
* - {py:obj}`FastTSPPostProcessor <src.policies.post_processing.FastTSPPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.FastTSPPostProcessor
    :summary:
    ```
* - {py:obj}`ClassicalLocalSearchPostProcessor <src.policies.post_processing.ClassicalLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.ClassicalLocalSearchPostProcessor
    :summary:
    ```
* - {py:obj}`RandomLocalSearchPostProcessor <src.policies.post_processing.RandomLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.RandomLocalSearchPostProcessor
    :summary:
    ```
* - {py:obj}`PathPostProcessor <src.policies.post_processing.PathPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.PathPostProcessor
    :summary:
    ```
* - {py:obj}`IteratedLocalSearchPostProcessor <src.policies.post_processing.IteratedLocalSearchPostProcessor>`
  - ```{autodoc2-docstring} src.policies.post_processing.IteratedLocalSearchPostProcessor
    :summary:
    ```
````

### API

`````{py:class} IPostProcessor
:canonical: src.policies.post_processing.IPostProcessor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.post_processing.IPostProcessor
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.IPostProcessor.process
:abstractmethod:

```{autodoc2-docstring} src.policies.post_processing.IPostProcessor.process
```

````

`````

`````{py:class} PostProcessorRegistry
:canonical: src.policies.post_processing.PostProcessorRegistry

```{autodoc2-docstring} src.policies.post_processing.PostProcessorRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.post_processing.PostProcessorRegistry._strategies
:type: typing.Dict[str, typing.Type[src.policies.post_processing.IPostProcessor]]
:value: >
   None

```{autodoc2-docstring} src.policies.post_processing.PostProcessorRegistry._strategies
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.post_processing.PostProcessorRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.post_processing.PostProcessorRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[src.policies.post_processing.IPostProcessor]]
:canonical: src.policies.post_processing.PostProcessorRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.post_processing.PostProcessorRegistry.get
```

````

`````

`````{py:class} PostProcessorFactory
:canonical: src.policies.post_processing.PostProcessorFactory

```{autodoc2-docstring} src.policies.post_processing.PostProcessorFactory
```

````{py:method} create(name: str) -> src.policies.post_processing.IPostProcessor
:canonical: src.policies.post_processing.PostProcessorFactory.create
:staticmethod:

```{autodoc2-docstring} src.policies.post_processing.PostProcessorFactory.create
```

````

`````

`````{py:class} FastTSPPostProcessor
:canonical: src.policies.post_processing.FastTSPPostProcessor

Bases: {py:obj}`src.policies.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.post_processing.FastTSPPostProcessor
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.FastTSPPostProcessor.process

```{autodoc2-docstring} src.policies.post_processing.FastTSPPostProcessor.process
```

````

`````

`````{py:class} ClassicalLocalSearchPostProcessor(operator_name: str = '2opt')
:canonical: src.policies.post_processing.ClassicalLocalSearchPostProcessor

Bases: {py:obj}`src.policies.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.post_processing.ClassicalLocalSearchPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.post_processing.ClassicalLocalSearchPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.ClassicalLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.post_processing.ClassicalLocalSearchPostProcessor.process
```

````

`````

`````{py:class} RandomLocalSearchPostProcessor
:canonical: src.policies.post_processing.RandomLocalSearchPostProcessor

Bases: {py:obj}`src.policies.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.post_processing.RandomLocalSearchPostProcessor
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.RandomLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.post_processing.RandomLocalSearchPostProcessor.process
```

````

`````

`````{py:class} PathPostProcessor
:canonical: src.policies.post_processing.PathPostProcessor

Bases: {py:obj}`src.policies.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.post_processing.PathPostProcessor
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.PathPostProcessor.process

```{autodoc2-docstring} src.policies.post_processing.PathPostProcessor.process
```

````

`````

`````{py:class} IteratedLocalSearchPostProcessor(ls_operator: typing.Union[str, typing.Dict[str, float]] = '2opt', perturbation_type: typing.Union[str, typing.Dict[str, float]] = 'double_bridge', n_restarts: int = 5, ls_iterations: int = 50, perturbation_strength: float = 0.2)
:canonical: src.policies.post_processing.IteratedLocalSearchPostProcessor

Bases: {py:obj}`src.policies.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.post_processing.IteratedLocalSearchPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.post_processing.IteratedLocalSearchPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.post_processing.IteratedLocalSearchPostProcessor.process

```{autodoc2-docstring} src.policies.post_processing.IteratedLocalSearchPostProcessor.process
```

````

`````
