# {py:mod}`src.policies.last_minute`

```{py:module} src.policies.last_minute
```

```{autodoc2-docstring} src.policies.last_minute
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LastMinutePolicy <src.policies.last_minute.LastMinutePolicy>`
  - ```{autodoc2-docstring} src.policies.last_minute.LastMinutePolicy
    :summary:
    ```
* - {py:obj}`ProfitPolicy <src.policies.last_minute.ProfitPolicy>`
  - ```{autodoc2-docstring} src.policies.last_minute.ProfitPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`policy_last_minute <src.policies.last_minute.policy_last_minute>`
  - ```{autodoc2-docstring} src.policies.last_minute.policy_last_minute
    :summary:
    ```
* - {py:obj}`policy_last_minute_and_path <src.policies.last_minute.policy_last_minute_and_path>`
  - ```{autodoc2-docstring} src.policies.last_minute.policy_last_minute_and_path
    :summary:
    ```
* - {py:obj}`policy_profit_reactive <src.policies.last_minute.policy_profit_reactive>`
  - ```{autodoc2-docstring} src.policies.last_minute.policy_profit_reactive
    :summary:
    ```
````

### API

````{py:function} policy_last_minute(bins: numpy.typing.NDArray[numpy.float64], distancesC: numpy.typing.NDArray[numpy.int32], lvl: numpy.typing.NDArray[numpy.float64], waste_type: str = 'plastic', area: str = 'riomaior', n_vehicles: int = 1, coords: pandas.DataFrame = None)
:canonical: src.policies.last_minute.policy_last_minute

```{autodoc2-docstring} src.policies.last_minute.policy_last_minute
```
````

````{py:function} policy_last_minute_and_path(bins: numpy.typing.NDArray[numpy.float64], distancesC: numpy.typing.NDArray[numpy.int32], paths_between_states: typing.List[typing.List[int]], lvl: numpy.typing.NDArray[numpy.float64], waste_type: str = 'plastic', area: str = 'riomaior', n_vehicles: int = 1, coords: pandas.DataFrame = None)
:canonical: src.policies.last_minute.policy_last_minute_and_path

```{autodoc2-docstring} src.policies.last_minute.policy_last_minute_and_path
```
````

````{py:function} policy_profit_reactive(bins: numpy.typing.NDArray[numpy.float64], distancesC: numpy.typing.NDArray[numpy.int32], waste_type: str = 'plastic', area: str = 'riomaior', n_vehicles: int = 1, coords: pandas.DataFrame = None, profit_threshold: float = 0.0)
:canonical: src.policies.last_minute.policy_profit_reactive

```{autodoc2-docstring} src.policies.last_minute.policy_profit_reactive
```
````

`````{py:class} LastMinutePolicy
:canonical: src.policies.last_minute.LastMinutePolicy

Bases: {py:obj}`src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.last_minute.LastMinutePolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.last_minute.LastMinutePolicy.execute

```{autodoc2-docstring} src.policies.last_minute.LastMinutePolicy.execute
```

````

`````

`````{py:class} ProfitPolicy
:canonical: src.policies.last_minute.ProfitPolicy

Bases: {py:obj}`src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.last_minute.ProfitPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.last_minute.ProfitPolicy.execute

```{autodoc2-docstring} src.policies.last_minute.ProfitPolicy.execute
```

````

`````
