# {py:mod}`src.interfaces.context.search_context`

```{py:module} src.interfaces.context.search_context
```

```{autodoc2-docstring} src.interfaces.context.search_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SearchPhase <src.interfaces.context.search_context.SearchPhase>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase
    :summary:
    ```
* - {py:obj}`SearchContext <src.interfaces.context.search_context.SearchContext>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`merge_context <src.interfaces.context.search_context.merge_context>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.merge_context
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionMetrics <src.interfaces.context.search_context.SelectionMetrics>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.SelectionMetrics
    :summary:
    ```
* - {py:obj}`ConstructionMetrics <src.interfaces.context.search_context.ConstructionMetrics>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.ConstructionMetrics
    :summary:
    ```
* - {py:obj}`AcceptanceMetrics <src.interfaces.context.search_context.AcceptanceMetrics>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.AcceptanceMetrics
    :summary:
    ```
* - {py:obj}`ImprovementMetrics <src.interfaces.context.search_context.ImprovementMetrics>`
  - ```{autodoc2-docstring} src.interfaces.context.search_context.ImprovementMetrics
    :summary:
    ```
````

### API

`````{py:class} SearchPhase()
:canonical: src.interfaces.context.search_context.SearchPhase

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase.__init__
```

````{py:attribute} SELECTION
:canonical: src.interfaces.context.search_context.SearchPhase.SELECTION
:value: >
   'selection'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase.SELECTION
```

````

````{py:attribute} CONSTRUCTION
:canonical: src.interfaces.context.search_context.SearchPhase.CONSTRUCTION
:value: >
   'construction'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase.CONSTRUCTION
```

````

````{py:attribute} IMPROVEMENT
:canonical: src.interfaces.context.search_context.SearchPhase.IMPROVEMENT
:value: >
   'improvement'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchPhase.IMPROVEMENT
```

````

`````

````{py:data} SelectionMetrics
:canonical: src.interfaces.context.search_context.SelectionMetrics
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.search_context.SelectionMetrics
```

````

````{py:data} ConstructionMetrics
:canonical: src.interfaces.context.search_context.ConstructionMetrics
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.search_context.ConstructionMetrics
```

````

````{py:data} AcceptanceMetrics
:canonical: src.interfaces.context.search_context.AcceptanceMetrics
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.search_context.AcceptanceMetrics
```

````

````{py:data} ImprovementMetrics
:canonical: src.interfaces.context.search_context.ImprovementMetrics
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.search_context.ImprovementMetrics
```

````

`````{py:class} SearchContext
:canonical: src.interfaces.context.search_context.SearchContext

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext
```

````{py:attribute} phase
:canonical: src.interfaces.context.search_context.SearchContext.phase
:type: src.interfaces.context.search_context.SearchPhase
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.phase
```

````

````{py:attribute} selection_metrics
:canonical: src.interfaces.context.search_context.SearchContext.selection_metrics
:type: src.interfaces.context.search_context.SelectionMetrics
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.selection_metrics
```

````

````{py:attribute} construction_metrics
:canonical: src.interfaces.context.search_context.SearchContext.construction_metrics
:type: src.interfaces.context.search_context.ConstructionMetrics
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.construction_metrics
```

````

````{py:attribute} acceptance_trace
:canonical: src.interfaces.context.search_context.SearchContext.acceptance_trace
:type: typing.List[src.interfaces.context.search_context.AcceptanceMetrics]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.acceptance_trace
```

````

````{py:attribute} improvement_metrics
:canonical: src.interfaces.context.search_context.SearchContext.improvement_metrics
:type: src.interfaces.context.search_context.ImprovementMetrics
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.improvement_metrics
```

````

````{py:attribute} metadata
:canonical: src.interfaces.context.search_context.SearchContext.metadata
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.metadata
```

````

````{py:method} initialize(selection_metrics: typing.Optional[src.interfaces.context.search_context.SelectionMetrics] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> src.interfaces.context.search_context.SearchContext
:canonical: src.interfaces.context.search_context.SearchContext.initialize
:classmethod:

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.initialize
```

````

````{py:method} merge(other: src.interfaces.context.search_context.SearchContext) -> src.interfaces.context.search_context.SearchContext
:canonical: src.interfaces.context.search_context.SearchContext.merge

```{autodoc2-docstring} src.interfaces.context.search_context.SearchContext.merge
```

````

`````

````{py:function} merge_context(ctx: src.interfaces.context.search_context.SearchContext, phase: typing.Optional[src.interfaces.context.search_context.SearchPhase] = None, selection_metrics: typing.Optional[src.interfaces.context.search_context.SelectionMetrics] = None, construction_metrics: typing.Optional[src.interfaces.context.search_context.ConstructionMetrics] = None, acceptance_metrics: typing.Optional[src.interfaces.context.search_context.AcceptanceMetrics] = None, improvement_metrics: typing.Optional[src.interfaces.context.search_context.ImprovementMetrics] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> src.interfaces.context.search_context.SearchContext
:canonical: src.interfaces.context.search_context.merge_context

```{autodoc2-docstring} src.interfaces.context.search_context.merge_context
```
````
