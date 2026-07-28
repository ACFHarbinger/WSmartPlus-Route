# {py:mod}`src.policies.helpers.operators.sequence_merging.sequential_selection`

```{py:module} src.policies.helpers.operators.sequence_merging.sequential_selection
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SsHhState <src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ss_hh_select <src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_select>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_select
    :summary:
    ```
* - {py:obj}`ss_hh_update <src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_update>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_update
    :summary:
    ```
* - {py:obj}`ss_hh_build_sequence <src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_build_sequence>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_build_sequence
    :summary:
    ```
* - {py:obj}`ss_hh_rank_operators <src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_rank_operators>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_rank_operators
    :summary:
    ```
* - {py:obj}`ss_hh_decay_scores <src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_decay_scores>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_decay_scores
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_UPDATE_ADDITIVE <src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_ADDITIVE>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_ADDITIVE
    :summary:
    ```
* - {py:obj}`_UPDATE_EMA <src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_EMA>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_EMA
    :summary:
    ```
* - {py:obj}`_UPDATE_SLIDING <src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_SLIDING>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_SLIDING
    :summary:
    ```
* - {py:obj}`_SELECT_GREEDY <src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_GREEDY>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_GREEDY
    :summary:
    ```
* - {py:obj}`_SELECT_EPSILON_GREEDY <src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_EPSILON_GREEDY>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_EPSILON_GREEDY
    :summary:
    ```
* - {py:obj}`_SELECT_SOFTMAX <src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_SOFTMAX>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_SOFTMAX
    :summary:
    ```
````

### API

````{py:data} _UPDATE_ADDITIVE
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_ADDITIVE
:value: >
   'additive'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_ADDITIVE
```

````

````{py:data} _UPDATE_EMA
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_EMA
:value: >
   'ema'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_EMA
```

````

````{py:data} _UPDATE_SLIDING
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_SLIDING
:value: >
   'sliding'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._UPDATE_SLIDING
```

````

````{py:data} _SELECT_GREEDY
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_GREEDY
:value: >
   'greedy'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_GREEDY
```

````

````{py:data} _SELECT_EPSILON_GREEDY
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_EPSILON_GREEDY
:value: >
   'epsilon_greedy'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_EPSILON_GREEDY
```

````

````{py:data} _SELECT_SOFTMAX
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_SOFTMAX
:value: >
   'softmax'

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection._SELECT_SOFTMAX
```

````

`````{py:class} SsHhState(op_names: typing.List[str], alpha_ema: float = 0.2, window_size: int = 10, update_strategy: str = _UPDATE_EMA, initial_score: float = 1.0, score_floor: float = 0.01)
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState.__init__
```

````{py:method} index(name: str) -> int
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState.index

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState.index
```

````

````{py:method} reset_scores(initial_score: float = 1.0) -> None
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState.reset_scores

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState.reset_scores
```

````

`````

````{py:function} ss_hh_select(state: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState, strategy: str = _SELECT_EPSILON_GREEDY, epsilon: float = 0.1, temperature: float = 1.0, rng: typing.Optional[random.Random] = None) -> str
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_select

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_select
```
````

````{py:function} ss_hh_update(state: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState, op_name: str, improvement: float) -> None
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_update

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_update
```
````

````{py:function} ss_hh_build_sequence(state: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState, seq_length: int, strategy: str = _SELECT_EPSILON_GREEDY, epsilon: float = 0.1, temperature: float = 1.0, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_build_sequence

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_build_sequence
```
````

````{py:function} ss_hh_rank_operators(state: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState) -> typing.List[tuple]
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_rank_operators

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_rank_operators
```
````

````{py:function} ss_hh_decay_scores(state: src.policies.helpers.operators.sequence_merging.sequential_selection.SsHhState, decay: float = 0.99) -> None
:canonical: src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_decay_scores

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequential_selection.ss_hh_decay_scores
```
````
