# {py:mod}`src.pipeline.rl.common.base.steps`

```{py:module} src.pipeline.rl.common.base.steps
```

```{autodoc2-docstring} src.pipeline.rl.common.base.steps
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepMixin <src.pipeline.rl.common.base.steps.StepMixin>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.base.steps.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.steps.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.base.steps.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.logger
```

````

`````{py:class} StepMixin()
:canonical: src.pipeline.rl.common.base.steps.StepMixin

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.__init__
```

````{py:method} _apply_must_go_selection(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.pipeline.rl.common.base.steps.StepMixin._apply_must_go_selection

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin._apply_must_go_selection
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.interfaces.env.IEnv] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.base.steps.StepMixin.calculate_loss
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.calculate_loss
```

````

````{py:method} shared_step(batch: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], batch_idx: int, phase: str) -> dict
:canonical: src.pipeline.rl.common.base.steps.StepMixin.shared_step

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.shared_step
```

````

````{py:method} training_step(batch: typing.Any, batch_idx: int) -> torch.Tensor
:canonical: src.pipeline.rl.common.base.steps.StepMixin.training_step

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.training_step
```

````

````{py:method} validation_step(batch: typing.Any, batch_idx: int) -> dict
:canonical: src.pipeline.rl.common.base.steps.StepMixin.validation_step

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.validation_step
```

````

````{py:method} test_step(batch: typing.Any, batch_idx: int) -> dict
:canonical: src.pipeline.rl.common.base.steps.StepMixin.test_step

```{autodoc2-docstring} src.pipeline.rl.common.base.steps.StepMixin.test_step
```

````

`````
