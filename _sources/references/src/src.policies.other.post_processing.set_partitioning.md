# {py:mod}`src.policies.other.post_processing.set_partitioning`

```{py:module} src.policies.other.post_processing.set_partitioning
```

```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SetPartitioningPostProcessor <src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_canonical <src.policies.other.post_processing.set_partitioning._canonical>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning._canonical
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.other.post_processing.set_partitioning.logger>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.other.post_processing.set_partitioning.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning.logger
```

````

````{py:function} _canonical(route: typing.List[int]) -> typing.Tuple[int, ...]
:canonical: src.policies.other.post_processing.set_partitioning._canonical

```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning._canonical
```
````

`````{py:class} SetPartitioningPostProcessor(**kwargs: typing.Any)
:canonical: src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor

Bases: {py:obj}`logic.src.interfaces.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.set_partitioning.SetPartitioningPostProcessor.process

````

`````
