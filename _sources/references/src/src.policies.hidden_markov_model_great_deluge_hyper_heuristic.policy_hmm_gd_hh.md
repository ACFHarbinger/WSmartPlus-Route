# {py:mod}`src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh`

```{py:module} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDHHPolicy <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy
    :summary:
    ```
````

### API

`````{py:class} HMMGDHHPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.hmm_gd_hh.HMMGDHHConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh.HMMGDHHPolicy._run_solver

````

`````
