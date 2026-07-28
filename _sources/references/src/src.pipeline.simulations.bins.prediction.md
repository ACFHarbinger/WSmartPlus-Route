# {py:mod}`src.pipeline.simulations.bins.prediction`

```{py:module} src.pipeline.simulations.bins.prediction
```

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScenarioTreeNode <src.pipeline.simulations.bins.prediction.ScenarioTreeNode>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode
    :summary:
    ```
* - {py:obj}`ScenarioTree <src.pipeline.simulations.bins.prediction.ScenarioTree>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree
    :summary:
    ```
* - {py:obj}`ScenarioGenerator <src.pipeline.simulations.bins.prediction.ScenarioGenerator>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`predict_days_to_overflow <src.pipeline.simulations.bins.prediction.predict_days_to_overflow>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.predict_days_to_overflow
    :summary:
    ```
* - {py:obj}`calculate_frequency_and_level <src.pipeline.simulations.bins.prediction.calculate_frequency_and_level>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.calculate_frequency_and_level
    :summary:
    ```
````

### API

`````{py:class} ScenarioTreeNode
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode
```

````{py:attribute} day
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode.day
:type: int
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode.day
```

````

````{py:attribute} wastes
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode.wastes
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode.wastes
```

````

````{py:attribute} probability
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode.probability
:type: float
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode.probability
```

````

````{py:attribute} children
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode.children
:type: typing.List[src.pipeline.simulations.bins.prediction.ScenarioTreeNode]
:value: >
   'field(...)'

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode.children
```

````

````{py:attribute} metadata
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTreeNode.metadata
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTreeNode.metadata
```

````

`````

`````{py:class} ScenarioTree
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTree

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree
```

````{py:attribute} root
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTree.root
:type: src.pipeline.simulations.bins.prediction.ScenarioTreeNode
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree.root
```

````

````{py:attribute} horizon
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTree.horizon
:type: int
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree.horizon
```

````

````{py:attribute} num_bins
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTree.num_bins
:type: int
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree.num_bins
```

````

````{py:method} get_scenarios_at_day(day: int) -> typing.List[src.pipeline.simulations.bins.prediction.ScenarioTreeNode]
:canonical: src.pipeline.simulations.bins.prediction.ScenarioTree.get_scenarios_at_day

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioTree.get_scenarios_at_day
```

````

`````

`````{py:class} ScenarioGenerator(method: str = 'stochastic', horizon: int = 7, seed: int = 42, distribution: str = 'mean', dist_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.pipeline.simulations.bins.prediction.ScenarioGenerator

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator.__init__
```

````{py:method} generate(current_wastes: numpy.ndarray, bin_stats: typing.Optional[typing.Dict[str, numpy.ndarray]] = None, truth_generator: typing.Optional[typing.Any] = None) -> src.pipeline.simulations.bins.prediction.ScenarioTree
:canonical: src.pipeline.simulations.bins.prediction.ScenarioGenerator.generate

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator.generate
```

````

````{py:method} _generate_oracle_path(root: src.pipeline.simulations.bins.prediction.ScenarioTreeNode, truth_generator: typing.Any) -> None
:canonical: src.pipeline.simulations.bins.prediction.ScenarioGenerator._generate_oracle_path

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator._generate_oracle_path
```

````

````{py:method} _generate_stochastic_tree(root: src.pipeline.simulations.bins.prediction.ScenarioTreeNode, bin_stats: typing.Optional[typing.Dict[str, numpy.ndarray]]) -> None
:canonical: src.pipeline.simulations.bins.prediction.ScenarioGenerator._generate_stochastic_tree

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.ScenarioGenerator._generate_stochastic_tree
```

````

`````

````{py:function} predict_days_to_overflow(ui: numpy.ndarray, vi: numpy.ndarray, f: numpy.ndarray, cl: float) -> numpy.ndarray
:canonical: src.pipeline.simulations.bins.prediction.predict_days_to_overflow

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.predict_days_to_overflow
```
````

````{py:function} calculate_frequency_and_level(ui: float, vi: float, cf: float) -> typing.Tuple[int, float]
:canonical: src.pipeline.simulations.bins.prediction.calculate_frequency_and_level

```{autodoc2-docstring} src.pipeline.simulations.bins.prediction.calculate_frequency_and_level
```
````
