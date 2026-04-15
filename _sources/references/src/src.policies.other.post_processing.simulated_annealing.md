# {py:mod}`src.policies.other.post_processing.simulated_annealing`

```{py:module} src.policies.other.post_processing.simulated_annealing
```

```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulatedAnnealingPostProcessor <src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor
    :summary:
    ```
````

### API

`````{py:class} SimulatedAnnealingPostProcessor(**kwargs: typing.Any)
:canonical: src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor

Bases: {py:obj}`logic.src.interfaces.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor.process

```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor.process
```

````

````{py:method} _random_move(routes: typing.List[typing.List[int]], dm: numpy.ndarray, rng: numpy.random.Generator, wastes: typing.Dict[int, float], capacity: float) -> typing.Tuple[typing.Any, float]
:canonical: src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor._random_move

```{autodoc2-docstring} src.policies.other.post_processing.simulated_annealing.SimulatedAnnealingPostProcessor._random_move
```

````

`````
