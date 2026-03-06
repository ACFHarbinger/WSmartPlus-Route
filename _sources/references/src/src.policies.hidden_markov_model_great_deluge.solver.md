# {py:mod}`src.policies.hidden_markov_model_great_deluge.solver`

```{py:module} src.policies.hidden_markov_model_great_deluge.solver
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDSolver <src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_STATE_IMPROVING <src.policies.hidden_markov_model_great_deluge.solver._STATE_IMPROVING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_IMPROVING
    :summary:
    ```
* - {py:obj}`_STATE_STAGNATING <src.policies.hidden_markov_model_great_deluge.solver._STATE_STAGNATING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_STAGNATING
    :summary:
    ```
* - {py:obj}`_STATE_ESCAPING <src.policies.hidden_markov_model_great_deluge.solver._STATE_ESCAPING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_ESCAPING
    :summary:
    ```
* - {py:obj}`_N_STATES <src.policies.hidden_markov_model_great_deluge.solver._N_STATES>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._N_STATES
    :summary:
    ```
````

### API

````{py:data} _STATE_IMPROVING
:canonical: src.policies.hidden_markov_model_great_deluge.solver._STATE_IMPROVING
:value: >
   0

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_IMPROVING
```

````

````{py:data} _STATE_STAGNATING
:canonical: src.policies.hidden_markov_model_great_deluge.solver._STATE_STAGNATING
:value: >
   1

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_STAGNATING
```

````

````{py:data} _STATE_ESCAPING
:canonical: src.policies.hidden_markov_model_great_deluge.solver._STATE_ESCAPING
:value: >
   2

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._STATE_ESCAPING
```

````

````{py:data} _N_STATES
:canonical: src.policies.hidden_markov_model_great_deluge.solver._N_STATES
:value: >
   3

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver._N_STATES
```

````

`````{py:class} HMMGDSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hidden_markov_model_great_deluge.params.HMMGDParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver.solve

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh0

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh1

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh2

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh3

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh4

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._llh4
```

````

````{py:method} _sample_llh(probs: numpy.ndarray) -> int
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._sample_llh

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._sample_llh
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._build_random_solution

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._build_random_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._evaluate

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._cost

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge.solver.HMMGDSolver._cost
```

````

`````
