# {py:mod}`src.policies.look_ahead`

```{py:module} src.policies.look_ahead
```

```{autodoc2-docstring} src.policies.look_ahead
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookAheadPolicy <src.policies.look_ahead.LookAheadPolicy>`
  - ```{autodoc2-docstring} src.policies.look_ahead.LookAheadPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`policy_lookahead <src.policies.look_ahead.policy_lookahead>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead
    :summary:
    ```
* - {py:obj}`policy_lookahead_vrpp <src.policies.look_ahead.policy_lookahead_vrpp>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_vrpp
    :summary:
    ```
* - {py:obj}`policy_lookahead_sans <src.policies.look_ahead.policy_lookahead_sans>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_sans
    :summary:
    ```
* - {py:obj}`policy_lookahead_hgs <src.policies.look_ahead.policy_lookahead_hgs>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_hgs
    :summary:
    ```
* - {py:obj}`policy_lookahead_alns <src.policies.look_ahead.policy_lookahead_alns>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_alns
    :summary:
    ```
* - {py:obj}`policy_lookahead_bcp <src.policies.look_ahead.policy_lookahead_bcp>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_bcp
    :summary:
    ```
* - {py:obj}`policy_lookahead_lk <src.policies.look_ahead.policy_lookahead_lk>`
  - ```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_lk
    :summary:
    ```
````

### API

````{py:function} policy_lookahead(binsids: typing.List[int], current_fill_levels: numpy.ndarray, accumulation_rates: numpy.ndarray, current_collection_day: int) -> typing.List[int]
:canonical: src.policies.look_ahead.policy_lookahead

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead
```
````

````{py:function} policy_lookahead_vrpp(current_fill_levels, binsids, must_go_bins, distance_matrix, values, number_vehicles=8, env=None, time_limit=600)
:canonical: src.policies.look_ahead.policy_lookahead_vrpp

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_vrpp
```
````

````{py:function} policy_lookahead_sans(data, bins_coordinates, distance_matrix, params, must_go_bins, values)
:canonical: src.policies.look_ahead.policy_lookahead_sans

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_sans
```
````

````{py:function} policy_lookahead_hgs(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords)
:canonical: src.policies.look_ahead.policy_lookahead_hgs

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_hgs
```
````

````{py:function} policy_lookahead_alns(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords, variant='custom')
:canonical: src.policies.look_ahead.policy_lookahead_alns

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_alns
```
````

````{py:function} policy_lookahead_bcp(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords, env=None)
:canonical: src.policies.look_ahead.policy_lookahead_bcp

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_bcp
```
````

````{py:function} policy_lookahead_lk(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords)
:canonical: src.policies.look_ahead.policy_lookahead_lk

```{autodoc2-docstring} src.policies.look_ahead.policy_lookahead_lk
```
````

`````{py:class} LookAheadPolicy
:canonical: src.policies.look_ahead.LookAheadPolicy

Bases: {py:obj}`src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.look_ahead.LookAheadPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.look_ahead.LookAheadPolicy.execute

```{autodoc2-docstring} src.policies.look_ahead.LookAheadPolicy.execute
```

````

`````
