# {py:mod}`src.pipeline.simulations.states.base.base`

```{py:module} src.pipeline.simulations.states.base.base
```

```{autodoc2-docstring} src.pipeline.simulations.states.base.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimState <src.pipeline.simulations.states.base.base.SimState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.base.base.SimState
    :summary:
    ```
````

### API

`````{py:class} SimState
:canonical: src.pipeline.simulations.states.base.base.SimState

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.states.base.base.SimState
```

````{py:attribute} context
:canonical: src.pipeline.simulations.states.base.base.SimState.context
:type: src.pipeline.simulations.states.base.context.SimulationContext
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.base.SimState.context
```

````

````{py:method} handle(ctx: src.pipeline.simulations.states.base.context.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.base.base.SimState.handle
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.states.base.base.SimState.handle
```

````

`````
