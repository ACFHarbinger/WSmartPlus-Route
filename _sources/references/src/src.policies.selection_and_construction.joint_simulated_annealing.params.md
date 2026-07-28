# {py:mod}`src.policies.selection_and_construction.joint_simulated_annealing.params`

```{py:module} src.policies.selection_and_construction.joint_simulated_annealing.params
```

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointSAParams <src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams
    :summary:
    ```
````

### API

`````{py:class} JointSAParams
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams
```

````{py:attribute} start_temp
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.start_temp
:type: float
:value: >
   1000.0

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.cooling_rate
```

````

````{py:attribute} max_steps
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.max_steps
:type: int
:value: >
   2000

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.max_steps
```

````

````{py:attribute} restart_limit
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.restart_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.restart_limit
```

````

````{py:attribute} prob_bit_flip
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.prob_bit_flip
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.prob_bit_flip
```

````

````{py:attribute} prob_route_swap
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.prob_route_swap
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.prob_route_swap
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.overflow_penalty
:type: float
:value: >
   1000.0

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.overflow_penalty
```

````

````{py:attribute} seed
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.seed
```

````

````{py:attribute} time_limit
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.time_limit
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams.from_config
```

````

`````
