# {py:mod}`src.policies.exact_stochastic_dynamic_programming.state_space`

```{py:module} src.policies.exact_stochastic_dynamic_programming.state_space
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StateSpaceManager <src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager
    :summary:
    ```
````

### API

`````{py:class} StateSpaceManager(num_nodes: int, discrete_levels: int, max_fill_rate: float)
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.__init__
```

````{py:method} get_all_states() -> typing.List[typing.Tuple[int, ...]]
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.get_all_states

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.get_all_states
```

````

````{py:method} _build_bin_transitions(mean_increment: float)
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager._build_bin_transitions

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager._build_bin_transitions
```

````

````{py:method} get_transition_probs(state: typing.Tuple[int, ...], action_set: typing.Union[frozenset, set]) -> typing.List[typing.Tuple[typing.Tuple[int, ...], float]]
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.get_transition_probs

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.get_transition_probs
```

````

````{py:method} state_to_fraction(state: typing.Tuple[int, ...]) -> typing.Dict[int, float]
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.state_to_fraction

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.state_to_fraction
```

````

````{py:method} fraction_to_state(fractions: typing.Dict[int, float]) -> typing.Tuple[int, ...]
:canonical: src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.fraction_to_state

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.state_space.StateSpaceManager.fraction_to_state
```

````

`````
