# {py:mod}`src.interfaces.context.multi_day_context`

```{py:module} src.interfaces.context.multi_day_context
```

```{autodoc2-docstring} src.interfaces.context.multi_day_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiDayContext <src.interfaces.context.multi_day_context.MultiDayContext>`
  - ```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext
    :summary:
    ```
````

### API

`````{py:class} MultiDayContext
:canonical: src.interfaces.context.multi_day_context.MultiDayContext

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext
```

````{py:attribute} day_index
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.day_index
:type: int
:value: >
   0

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.day_index
```

````

````{py:attribute} previous_days_metadata
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.previous_days_metadata
:type: typing.List[typing.Dict[str, typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.previous_days_metadata
```

````

````{py:attribute} full_plan_snapshot
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.full_plan_snapshot
:type: typing.Optional[typing.List[typing.List[typing.List[int]]]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.full_plan_snapshot
```

````

````{py:attribute} accumulated_stats
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.accumulated_stats
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.accumulated_stats
```

````

````{py:attribute} extra
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.extra
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.extra
```

````

````{py:method} initialize(day_index: int = 0) -> src.interfaces.context.multi_day_context.MultiDayContext
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.initialize
:classmethod:

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.initialize
```

````

````{py:method} update(**patch: typing.Any) -> src.interfaces.context.multi_day_context.MultiDayContext
:canonical: src.interfaces.context.multi_day_context.MultiDayContext.update

```{autodoc2-docstring} src.interfaces.context.multi_day_context.MultiDayContext.update
```

````

`````
