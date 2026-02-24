# {py:mod}`src.tracking.viz_mixin`

```{py:module} src.tracking.viz_mixin
```

```{autodoc2-docstring} src.tracking.viz_mixin
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyStateRecorder <src.tracking.viz_mixin.PolicyStateRecorder>`
  - ```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder
    :summary:
    ```
* - {py:obj}`PolicyVizMixin <src.tracking.viz_mixin.PolicyVizMixin>`
  - ```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin
    :summary:
    ```
````

### API

`````{py:class} PolicyStateRecorder(max_history: int = 5000)
:canonical: src.tracking.viz_mixin.PolicyStateRecorder

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder.__init__
```

````{py:method} record(**kwargs: typing.Any) -> None
:canonical: src.tracking.viz_mixin.PolicyStateRecorder.record

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder.record
```

````

````{py:method} get() -> typing.Dict[str, typing.List[typing.Any]]
:canonical: src.tracking.viz_mixin.PolicyStateRecorder.get

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder.get
```

````

````{py:method} reset() -> None
:canonical: src.tracking.viz_mixin.PolicyStateRecorder.reset

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder.reset
```

````

````{py:method} __len__() -> int
:canonical: src.tracking.viz_mixin.PolicyStateRecorder.__len__

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyStateRecorder.__len__
```

````

`````

`````{py:class} PolicyVizMixin
:canonical: src.tracking.viz_mixin.PolicyVizMixin

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin
```

````{py:property} _viz
:canonical: src.tracking.viz_mixin.PolicyVizMixin._viz
:type: src.tracking.viz_mixin.PolicyStateRecorder

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin._viz
```

````

````{py:method} _viz_record(**kwargs: typing.Any) -> None
:canonical: src.tracking.viz_mixin.PolicyVizMixin._viz_record

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin._viz_record
```

````

````{py:method} get_viz_data() -> typing.Dict[str, typing.List[typing.Any]]
:canonical: src.tracking.viz_mixin.PolicyVizMixin.get_viz_data

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin.get_viz_data
```

````

````{py:method} reset_viz() -> None
:canonical: src.tracking.viz_mixin.PolicyVizMixin.reset_viz

```{autodoc2-docstring} src.tracking.viz_mixin.PolicyVizMixin.reset_viz
```

````

`````
