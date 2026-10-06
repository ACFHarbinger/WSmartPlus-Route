# {py:mod}`src.pipeline.simulations.actions.base`

```{py:module} src.pipeline.simulations.actions.base
```

```{autodoc2-docstring} src.pipeline.simulations.actions.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_AttrDict <src.pipeline.simulations.actions.base._AttrDict>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._AttrDict
    :summary:
    ```
* - {py:obj}`SimulationAction <src.pipeline.simulations.actions.base.SimulationAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_find_key <src.pipeline.simulations.actions.base._find_key>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._find_key
    :summary:
    ```
* - {py:obj}`_flatten_config <src.pipeline.simulations.actions.base._flatten_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._flatten_config
    :summary:
    ```
* - {py:obj}`_is_mapping <src.pipeline.simulations.actions.base._is_mapping>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._is_mapping
    :summary:
    ```
* - {py:obj}`_as_plain_dict <src.pipeline.simulations.actions.base._as_plain_dict>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._as_plain_dict
    :summary:
    ```
* - {py:obj}`_is_non_string_sequence <src.pipeline.simulations.actions.base._is_non_string_sequence>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._is_non_string_sequence
    :summary:
    ```
* - {py:obj}`_unwrap_variant_value <src.pipeline.simulations.actions.base._unwrap_variant_value>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._unwrap_variant_value
    :summary:
    ```
* - {py:obj}`_file_yaml_pairs <src.pipeline.simulations.actions.base._file_yaml_pairs>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._file_yaml_pairs
    :summary:
    ```
* - {py:obj}`_load_policy_yaml_section <src.pipeline.simulations.actions.base._load_policy_yaml_section>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._load_policy_yaml_section
    :summary:
    ```
* - {py:obj}`_params_as_dict <src.pipeline.simulations.actions.base._params_as_dict>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._params_as_dict
    :summary:
    ```
* - {py:obj}`_make_acceptance_config <src.pipeline.simulations.actions.base._make_acceptance_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._make_acceptance_config
    :summary:
    ```
* - {py:obj}`_acceptance_config_from_entry <src.pipeline.simulations.actions.base._acceptance_config_from_entry>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._acceptance_config_from_entry
    :summary:
    ```
* - {py:obj}`_inject_acceptance_into_config <src.pipeline.simulations.actions.base._inject_acceptance_into_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._inject_acceptance_into_config
    :summary:
    ```
* - {py:obj}`_live_ac_payload <src.pipeline.simulations.actions.base._live_ac_payload>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._live_ac_payload
    :summary:
    ```
* - {py:obj}`_attrify <src.pipeline.simulations.actions.base._attrify>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._attrify
    :summary:
    ```
* - {py:obj}`_ensure_mutable_config <src.pipeline.simulations.actions.base._ensure_mutable_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._ensure_mutable_config
    :summary:
    ```
* - {py:obj}`_pop_context_key <src.pipeline.simulations.actions.base._pop_context_key>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._pop_context_key
    :summary:
    ```
* - {py:obj}`_append_context_list <src.pipeline.simulations.actions.base._append_context_list>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._append_context_list
    :summary:
    ```
* - {py:obj}`_jsonable <src.pipeline.simulations.actions.base._jsonable>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._jsonable
    :summary:
    ```
* - {py:obj}`_constructor_from_policy_id <src.pipeline.simulations.actions.base._constructor_from_policy_id>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._constructor_from_policy_id
    :summary:
    ```
* - {py:obj}`_set_live_capture_meta <src.pipeline.simulations.actions.base._set_live_capture_meta>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._set_live_capture_meta
    :summary:
    ```
* - {py:obj}`_record_live_params <src.pipeline.simulations.actions.base._record_live_params>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._record_live_params
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_EXPANDED_KEYS <src.pipeline.simulations.actions.base._EXPANDED_KEYS>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._EXPANDED_KEYS
    :summary:
    ```
* - {py:obj}`_LIVE_CAPTURE_META <src.pipeline.simulations.actions.base._LIVE_CAPTURE_META>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._LIVE_CAPTURE_META
    :summary:
    ```
* - {py:obj}`_PAPER_CONSTRUCTORS <src.pipeline.simulations.actions.base._PAPER_CONSTRUCTORS>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._PAPER_CONSTRUCTORS
    :summary:
    ```
````

### API

````{py:function} _find_key(d: typing.Any, target_key: str) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._find_key

```{autodoc2-docstring} src.pipeline.simulations.actions.base._find_key
```
````

````{py:data} _EXPANDED_KEYS
:canonical: src.pipeline.simulations.actions.base._EXPANDED_KEYS
:value: >
   ('mandatory_selection', 'route_improvement', 'acceptance_criteria', 'acceptance_criterion')

```{autodoc2-docstring} src.pipeline.simulations.actions.base._EXPANDED_KEYS
```

````

````{py:function} _flatten_config(cfg: typing.Any) -> dict
:canonical: src.pipeline.simulations.actions.base._flatten_config

```{autodoc2-docstring} src.pipeline.simulations.actions.base._flatten_config
```
````

````{py:function} _is_mapping(obj: typing.Any) -> bool
:canonical: src.pipeline.simulations.actions.base._is_mapping

```{autodoc2-docstring} src.pipeline.simulations.actions.base._is_mapping
```
````

````{py:function} _as_plain_dict(cfg: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.actions.base._as_plain_dict

```{autodoc2-docstring} src.pipeline.simulations.actions.base._as_plain_dict
```
````

````{py:function} _is_non_string_sequence(obj: typing.Any) -> bool
:canonical: src.pipeline.simulations.actions.base._is_non_string_sequence

```{autodoc2-docstring} src.pipeline.simulations.actions.base._is_non_string_sequence
```
````

````{py:function} _unwrap_variant_value(val: typing.Any) -> str
:canonical: src.pipeline.simulations.actions.base._unwrap_variant_value

```{autodoc2-docstring} src.pipeline.simulations.actions.base._unwrap_variant_value
```
````

````{py:function} _file_yaml_pairs(item: typing.Any) -> typing.List[typing.Tuple[str, str]]
:canonical: src.pipeline.simulations.actions.base._file_yaml_pairs

```{autodoc2-docstring} src.pipeline.simulations.actions.base._file_yaml_pairs
```
````

````{py:function} _load_policy_yaml_section(file_key: str, variant: str) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.actions.base._load_policy_yaml_section

```{autodoc2-docstring} src.pipeline.simulations.actions.base._load_policy_yaml_section
```
````

````{py:function} _params_as_dict(params: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.actions.base._params_as_dict

```{autodoc2-docstring} src.pipeline.simulations.actions.base._params_as_dict
```
````

````{py:function} _make_acceptance_config(method: str, params: typing.Any) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._make_acceptance_config

```{autodoc2-docstring} src.pipeline.simulations.actions.base._make_acceptance_config
```
````

````{py:function} _acceptance_config_from_entry(raw: typing.Any) -> typing.Optional[typing.Any]
:canonical: src.pipeline.simulations.actions.base._acceptance_config_from_entry

```{autodoc2-docstring} src.pipeline.simulations.actions.base._acceptance_config_from_entry
```
````

````{py:function} _inject_acceptance_into_config(cfg: typing.Any) -> typing.Optional[typing.Any]
:canonical: src.pipeline.simulations.actions.base._inject_acceptance_into_config

```{autodoc2-docstring} src.pipeline.simulations.actions.base._inject_acceptance_into_config
```
````

````{py:function} _live_ac_payload(resolved: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.actions.base._live_ac_payload

```{autodoc2-docstring} src.pipeline.simulations.actions.base._live_ac_payload
```
````

`````{py:class} _AttrDict()
:canonical: src.pipeline.simulations.actions.base._AttrDict

Bases: {py:obj}`dict`

```{autodoc2-docstring} src.pipeline.simulations.actions.base._AttrDict
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.actions.base._AttrDict.__init__
```

````{py:method} __getattr__(name: str) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._AttrDict.__getattr__

```{autodoc2-docstring} src.pipeline.simulations.actions.base._AttrDict.__getattr__
```

````

````{py:method} __setattr__(name: str, value: typing.Any) -> None
:canonical: src.pipeline.simulations.actions.base._AttrDict.__setattr__

````

`````

````{py:function} _attrify(obj: typing.Any) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._attrify

```{autodoc2-docstring} src.pipeline.simulations.actions.base._attrify
```
````

````{py:function} _ensure_mutable_config(context: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._ensure_mutable_config

```{autodoc2-docstring} src.pipeline.simulations.actions.base._ensure_mutable_config
```
````

````{py:function} _pop_context_key(context: typing.Any, key: str) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._pop_context_key

```{autodoc2-docstring} src.pipeline.simulations.actions.base._pop_context_key
```
````

````{py:function} _append_context_list(context: typing.Any, key: str, item: typing.Any) -> None
:canonical: src.pipeline.simulations.actions.base._append_context_list

```{autodoc2-docstring} src.pipeline.simulations.actions.base._append_context_list
```
````

````{py:function} _jsonable(obj: typing.Any) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._jsonable

```{autodoc2-docstring} src.pipeline.simulations.actions.base._jsonable
```
````

````{py:data} _LIVE_CAPTURE_META
:canonical: src.pipeline.simulations.actions.base._LIVE_CAPTURE_META
:type: contextvars.ContextVar[typing.Optional[typing.Dict[str, typing.Any]]]
:value: >
   'ContextVar(...)'

```{autodoc2-docstring} src.pipeline.simulations.actions.base._LIVE_CAPTURE_META
```

````

````{py:data} _PAPER_CONSTRUCTORS
:canonical: src.pipeline.simulations.actions.base._PAPER_CONSTRUCTORS
:value: >
   ('ms_bpc_sp', 'pg_clns', 'swc_tcf', 'aco_hh', 'psoma', 'alns', 'hgs', 'bpc', 'na')

```{autodoc2-docstring} src.pipeline.simulations.actions.base._PAPER_CONSTRUCTORS
```

````

````{py:function} _constructor_from_policy_id(policy_id: str) -> str
:canonical: src.pipeline.simulations.actions.base._constructor_from_policy_id

```{autodoc2-docstring} src.pipeline.simulations.actions.base._constructor_from_policy_id
```
````

````{py:function} _set_live_capture_meta(context: typing.Any) -> None
:canonical: src.pipeline.simulations.actions.base._set_live_capture_meta

```{autodoc2-docstring} src.pipeline.simulations.actions.base._set_live_capture_meta
```
````

````{py:function} _record_live_params(kind: str, payload: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.base._record_live_params

```{autodoc2-docstring} src.pipeline.simulations.actions.base._record_live_params
```
````

`````{py:class} SimulationAction
:canonical: src.pipeline.simulations.actions.base.SimulationAction

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.base.SimulationAction.execute
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction.execute
```

````

`````
