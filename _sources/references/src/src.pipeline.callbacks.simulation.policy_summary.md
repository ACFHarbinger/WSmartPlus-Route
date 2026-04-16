# {py:mod}`src.pipeline.callbacks.simulation.policy_summary`

```{py:module} src.pipeline.callbacks.simulation.policy_summary
```

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicySummaryCallback <src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback
    :summary:
    ```
````

### API

`````{py:class} PolicySummaryCallback
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback
```

````{py:method} display(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback.display

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback.display
```

````

````{py:method} _extract_engine(policy_name: str, config: typing.Dict[str, typing.Any]) -> str
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_engine

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_engine
```

````

````{py:method} _extract_selection(config: typing.Dict[str, typing.Any]) -> str
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_selection

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_selection
```

````

````{py:method} _parse_selection_item(item: typing.Any) -> tuple[str, str]
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_selection_item

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_selection_item
```

````

````{py:method} _parse_mandatory_config_params(config: logic.src.configs.MandatorySelectionConfig) -> str
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_mandatory_config_params

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_mandatory_config_params
```

````

````{py:method} _parse_traversable_params(item: logic.src.interfaces.ITraversable) -> str
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_traversable_params

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._parse_traversable_params
```

````

````{py:method} _extract_route_improvement(config: typing.Dict[str, typing.Any]) -> str
:canonical: src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_route_improvement

```{autodoc2-docstring} src.pipeline.callbacks.simulation.policy_summary.PolicySummaryCallback._extract_route_improvement
```

````

`````
