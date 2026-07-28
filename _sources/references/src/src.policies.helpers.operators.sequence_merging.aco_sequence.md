# {py:mod}`src.policies.helpers.operators.sequence_merging.aco_sequence`

```{py:module} src.policies.helpers.operators.sequence_merging.aco_sequence
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AcoSequenceState <src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`aco_build_sequence <src.policies.helpers.operators.sequence_merging.aco_sequence.aco_build_sequence>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_build_sequence
    :summary:
    ```
* - {py:obj}`aco_update_pheromones <src.policies.helpers.operators.sequence_merging.aco_sequence.aco_update_pheromones>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_update_pheromones
    :summary:
    ```
* - {py:obj}`aco_best_sequence <src.policies.helpers.operators.sequence_merging.aco_sequence.aco_best_sequence>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_best_sequence
    :summary:
    ```
* - {py:obj}`_acs_transition <src.policies.helpers.operators.sequence_merging.aco_sequence._acs_transition>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence._acs_transition
    :summary:
    ```
````

### API

`````{py:class} AcoSequenceState(op_names: typing.List[str], alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, Q: float = 1.0, tau_min: float = 0.01, tau_max: float = 10.0, tau_init: float = 1.0)
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState.__init__
```

````{py:method} index(name: str) -> int
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState.index

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState.index
```

````

`````

````{py:function} aco_build_sequence(state: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState, seq_length: int, start_op: typing.Optional[str] = None, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence.aco_build_sequence

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_build_sequence
```
````

````{py:function} aco_update_pheromones(state: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState, sequence: typing.List[str], improvement: float, evaporate_all: bool = True) -> None
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence.aco_update_pheromones

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_update_pheromones
```
````

````{py:function} aco_best_sequence(state: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState, seq_length: int, start_op: typing.Optional[str] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence.aco_best_sequence

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence.aco_best_sequence
```
````

````{py:function} _acs_transition(state: src.policies.helpers.operators.sequence_merging.aco_sequence.AcoSequenceState, current: str, rng: random.Random, exploit_prob: float = 0.9) -> str
:canonical: src.policies.helpers.operators.sequence_merging.aco_sequence._acs_transition

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.aco_sequence._acs_transition
```
````
