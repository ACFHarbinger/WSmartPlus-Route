# {py:mod}`src.pipeline.simulations.day`

```{py:module} src.pipeline.simulations.day
```

```{autodoc2-docstring} src.pipeline.simulations.day
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`set_daily_waste <src.pipeline.simulations.day.set_daily_waste>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day.set_daily_waste
    :summary:
    ```
* - {py:obj}`get_daily_results <src.pipeline.simulations.day.get_daily_results>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day.get_daily_results
    :summary:
    ```
* - {py:obj}`send_daily_output_to_gui <src.pipeline.simulations.day.send_daily_output_to_gui>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day.send_daily_output_to_gui
    :summary:
    ```
* - {py:obj}`run_day <src.pipeline.simulations.day.run_day>`
  - ```{autodoc2-docstring} src.pipeline.simulations.day.run_day
    :summary:
    ```
````

### API

````{py:function} set_daily_waste(model_data: typing.Dict[str, typing.Any], waste: numpy.ndarray, device: torch.device, fill: typing.Optional[numpy.ndarray] = None) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.day.set_daily_waste

```{autodoc2-docstring} src.pipeline.simulations.day.set_daily_waste
```
````

````{py:function} get_daily_results(total_collected: float, ncol: int, cost: float, tour: typing.List[int], day: int, new_overflows: int, sum_lost: float, coordinates: pandas.DataFrame, profit: float) -> typing.Dict[str, typing.Union[int, float, typing.List[typing.Union[int, str]]]]
:canonical: src.pipeline.simulations.day.get_daily_results

```{autodoc2-docstring} src.pipeline.simulations.day.get_daily_results
```
````

````{py:function} send_daily_output_to_gui(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.pipeline.simulations.day.send_daily_output_to_gui

```{autodoc2-docstring} src.pipeline.simulations.day.send_daily_output_to_gui
```
````

````{py:function} run_day(context: logic.src.pipeline.simulations.context.SimulationDayContext) -> logic.src.pipeline.simulations.context.SimulationDayContext
:canonical: src.pipeline.simulations.day.run_day

```{autodoc2-docstring} src.pipeline.simulations.day.run_day
```
````
