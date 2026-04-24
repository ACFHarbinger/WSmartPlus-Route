# {py:mod}`src.policies.mandatory_selection.selection_dispatcher_thompson`

```{py:module} src.policies.mandatory_selection.selection_dispatcher_thompson
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThompsonDispatcher <src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.mandatory_selection.selection_dispatcher_thompson.logger>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.logger
```

````

`````{py:class} ThompsonDispatcher
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher
```

````{py:attribute} _lock
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._lock
:value: >
   'Lock(...)'

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._lock
```

````

````{py:attribute} _shared_state
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._shared_state
:type: typing.Dict[str, typing.Tuple[float, float]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._shared_state
```

````

````{py:method} record_reward(strategy_name: str, reward: float, baseline: typing.Optional[float] = None, success_prob: typing.Optional[float] = None, state_path: typing.Optional[str] = None) -> None
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher.record_reward
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher.record_reward
```

````

````{py:method} _load_state_locked(path: typing.Optional[str]) -> typing.Dict[str, typing.Tuple[float, float]]
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._load_state_locked
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._load_state_locked
```

````

````{py:method} _save_state_locked(state: typing.Dict[str, typing.Tuple[float, float]], path: typing.Optional[str]) -> None
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._save_state_locked
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher._save_state_locked
```

````

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_dispatcher_thompson.ThompsonDispatcher.select_bins
```

````

`````
