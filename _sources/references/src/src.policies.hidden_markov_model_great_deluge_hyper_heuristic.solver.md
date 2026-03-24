# {py:mod}`src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver`

```{py:module} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDHHSolver <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_STATE_IMPROVING <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_IMPROVING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_IMPROVING
    :summary:
    ```
* - {py:obj}`_STATE_STAGNATING <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_STAGNATING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_STAGNATING
    :summary:
    ```
* - {py:obj}`_STATE_ESCAPING <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_ESCAPING>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_ESCAPING
    :summary:
    ```
* - {py:obj}`_N_STATES <src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_STATES>`
  - ```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_STATES
    :summary:
    ```
````

### API

````{py:data} _STATE_IMPROVING
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_IMPROVING
:value: >
   0

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_IMPROVING
```

````

````{py:data} _STATE_STAGNATING
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_STAGNATING
:value: >
   1

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_STAGNATING
```

````

````{py:data} _STATE_ESCAPING
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_ESCAPING
:value: >
   2

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._STATE_ESCAPING
```

````

````{py:data} _N_STATES
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_STATES
:value: >
   3

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_STATES
```

````

`````{py:class} HMMGDHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.solve

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.solve
```

````

````{py:method} _gaussian_pdf(delta_norm: float) -> numpy.ndarray
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._gaussian_pdf

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._gaussian_pdf
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh0

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh1

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh2

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh3

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh4

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh4
```

````

````{py:method} _llh5(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh5

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh5
```

````

````{py:method} _llh6(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh6

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh6
```

````

````{py:method} _sample_llh(probs: numpy.ndarray) -> int
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._sample_llh

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._sample_llh
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._build_random_solution

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._build_random_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._evaluate

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._cost

```{autodoc2-docstring} src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._cost
```

````

`````
