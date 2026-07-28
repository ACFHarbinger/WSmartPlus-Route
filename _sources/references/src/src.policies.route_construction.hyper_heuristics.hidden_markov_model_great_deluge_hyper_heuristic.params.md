# {py:mod}`src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params`

```{py:module} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDHHParams <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams
    :summary:
    ```
````

### API

`````{py:class} HMMGDHHParams
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams
```

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.max_iterations
```

````

````{py:attribute} flood_margin
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.flood_margin
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.flood_margin
```

````

````{py:attribute} rain_speed
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.rain_speed
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.rain_speed
```

````

````{py:attribute} learning_rate
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.learning_rate
```

````

````{py:attribute} n_removal
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.n_llh
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams.from_config
```

````

`````
