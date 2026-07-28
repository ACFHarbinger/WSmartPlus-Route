# {py:mod}`src.policies.helpers.operators.sequence_merging.markov_chain_sequence`

```{py:module} src.policies.helpers.operators.sequence_merging.markov_chain_sequence
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MarkovSequenceState <src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`markov_sample_sequence <src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_sample_sequence>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_sample_sequence
    :summary:
    ```
* - {py:obj}`markov_update <src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_update>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_update
    :summary:
    ```
* - {py:obj}`markov_fit_from_log <src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_fit_from_log>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_fit_from_log
    :summary:
    ```
* - {py:obj}`markov_stationary_distribution <src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_stationary_distribution>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_stationary_distribution
    :summary:
    ```
````

### API

`````{py:class} MarkovSequenceState(op_names: typing.List[str], alpha_ema: float = 0.1, allow_self_loops: bool = False)
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState.__init__
```

````{py:method} _normalise() -> None
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState._normalise

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState._normalise
```

````

````{py:method} index(name: str) -> int
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState.index

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState.index
```

````

`````

````{py:function} markov_sample_sequence(state: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState, seq_length: int, start_op: typing.Optional[str] = None, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_sample_sequence

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_sample_sequence
```
````

````{py:function} markov_update(state: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState, sequence: typing.List[str], improvement: float, reward_threshold: float = 0.0) -> None
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_update

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_update
```
````

````{py:function} markov_fit_from_log(state: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState, execution_log: typing.List[typing.List[str]]) -> None
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_fit_from_log

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_fit_from_log
```
````

````{py:function} markov_stationary_distribution(state: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.MarkovSequenceState) -> numpy.ndarray
:canonical: src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_stationary_distribution

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.markov_chain_sequence.markov_stationary_distribution
```
````
