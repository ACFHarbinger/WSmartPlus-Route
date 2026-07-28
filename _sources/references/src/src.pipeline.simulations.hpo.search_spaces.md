# {py:mod}`src.pipeline.simulations.hpo.search_spaces`

```{py:module} src.pipeline.simulations.hpo.search_spaces
```

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`validate_search_space <src.pipeline.simulations.hpo.search_spaces.validate_search_space>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.validate_search_space
    :summary:
    ```
* - {py:obj}`load_all_search_spaces <src.pipeline.simulations.hpo.search_spaces.load_all_search_spaces>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.load_all_search_spaces
    :summary:
    ```
* - {py:obj}`get_component_search_space <src.pipeline.simulations.hpo.search_spaces.get_component_search_space>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.get_component_search_space
    :summary:
    ```
* - {py:obj}`compose_search_space <src.pipeline.simulations.hpo.search_spaces.compose_search_space>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.compose_search_space
    :summary:
    ```
* - {py:obj}`get_search_space <src.pipeline.simulations.hpo.search_spaces.get_search_space>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.get_search_space
    :summary:
    ```
* - {py:obj}`_extract_params_from_config <src.pipeline.simulations.hpo.search_spaces._extract_params_from_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces._extract_params_from_config
    :summary:
    ```
* - {py:obj}`generate_policy_filters <src.pipeline.simulations.hpo.search_spaces.generate_policy_filters>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_policy_filters
    :summary:
    ```
* - {py:obj}`generate_route_improvement_interceptors <src.pipeline.simulations.hpo.search_spaces.generate_route_improvement_interceptors>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_route_improvement_interceptors
    :summary:
    ```
* - {py:obj}`generate_mandatory_selection_jobs <src.pipeline.simulations.hpo.search_spaces.generate_mandatory_selection_jobs>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_mandatory_selection_jobs
    :summary:
    ```
* - {py:obj}`generate_acceptance_criteria_rules <src.pipeline.simulations.hpo.search_spaces.generate_acceptance_criteria_rules>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_acceptance_criteria_rules
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILTERS_DIR <src.pipeline.simulations.hpo.search_spaces.FILTERS_DIR>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.FILTERS_DIR
    :summary:
    ```
* - {py:obj}`INTERCEPTORS_DIR <src.pipeline.simulations.hpo.search_spaces.INTERCEPTORS_DIR>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.INTERCEPTORS_DIR
    :summary:
    ```
* - {py:obj}`JOBS_DIR <src.pipeline.simulations.hpo.search_spaces.JOBS_DIR>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.JOBS_DIR
    :summary:
    ```
* - {py:obj}`RULES_DIR <src.pipeline.simulations.hpo.search_spaces.RULES_DIR>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.RULES_DIR
    :summary:
    ```
* - {py:obj}`_VALID_TYPES <src.pipeline.simulations.hpo.search_spaces._VALID_TYPES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces._VALID_TYPES
    :summary:
    ```
* - {py:obj}`FILTER_SPACES <src.pipeline.simulations.hpo.search_spaces.FILTER_SPACES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.FILTER_SPACES
    :summary:
    ```
* - {py:obj}`INTERCEPTOR_SPACES <src.pipeline.simulations.hpo.search_spaces.INTERCEPTOR_SPACES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.INTERCEPTOR_SPACES
    :summary:
    ```
* - {py:obj}`JOB_SPACES <src.pipeline.simulations.hpo.search_spaces.JOB_SPACES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.JOB_SPACES
    :summary:
    ```
* - {py:obj}`RULE_SPACES <src.pipeline.simulations.hpo.search_spaces.RULE_SPACES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.RULE_SPACES
    :summary:
    ```
* - {py:obj}`POLICY_SEARCH_SPACES <src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES
    :summary:
    ```
````

### API

````{py:data} FILTERS_DIR
:canonical: src.pipeline.simulations.hpo.search_spaces.FILTERS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.FILTERS_DIR
```

````

````{py:data} INTERCEPTORS_DIR
:canonical: src.pipeline.simulations.hpo.search_spaces.INTERCEPTORS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.INTERCEPTORS_DIR
```

````

````{py:data} JOBS_DIR
:canonical: src.pipeline.simulations.hpo.search_spaces.JOBS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.JOBS_DIR
```

````

````{py:data} RULES_DIR
:canonical: src.pipeline.simulations.hpo.search_spaces.RULES_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.RULES_DIR
```

````

````{py:data} _VALID_TYPES
:canonical: src.pipeline.simulations.hpo.search_spaces._VALID_TYPES
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces._VALID_TYPES
```

````

````{py:function} validate_search_space(space: typing.Dict[str, typing.Any], policy_name: str) -> typing.List[str]
:canonical: src.pipeline.simulations.hpo.search_spaces.validate_search_space

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.validate_search_space
```
````

````{py:function} load_all_search_spaces(directory: typing.Optional[str] = None) -> typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:canonical: src.pipeline.simulations.hpo.search_spaces.load_all_search_spaces

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.load_all_search_spaces
```
````

````{py:data} FILTER_SPACES
:canonical: src.pipeline.simulations.hpo.search_spaces.FILTER_SPACES
:type: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:value: >
   'load_all_search_spaces(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.FILTER_SPACES
```

````

````{py:data} INTERCEPTOR_SPACES
:canonical: src.pipeline.simulations.hpo.search_spaces.INTERCEPTOR_SPACES
:type: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:value: >
   'load_all_search_spaces(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.INTERCEPTOR_SPACES
```

````

````{py:data} JOB_SPACES
:canonical: src.pipeline.simulations.hpo.search_spaces.JOB_SPACES
:type: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:value: >
   'load_all_search_spaces(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.JOB_SPACES
```

````

````{py:data} RULE_SPACES
:canonical: src.pipeline.simulations.hpo.search_spaces.RULE_SPACES
:type: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:value: >
   'load_all_search_spaces(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.RULE_SPACES
```

````

````{py:data} POLICY_SEARCH_SPACES
:canonical: src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES
```

````

````{py:function} get_component_search_space(component_type: str, name: str, raise_on_invalid: bool = True, keywords: typing.Optional[str] = None, index: typing.Optional[int] = None) -> typing.Dict[str, typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.hpo.search_spaces.get_component_search_space

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.get_component_search_space
```
````

````{py:function} compose_search_space(job: typing.Optional[str] = None, filter: typing.Optional[typing.Union[str, typing.List[str]]] = None, interceptor: typing.Optional[typing.Union[str, typing.List[str]]] = None, rule: typing.Optional[typing.Union[str, typing.List[str]]] = None, job_keywords: typing.Optional[str] = None, filter_keywords: typing.Optional[str] = None, interceptor_keywords: typing.Optional[str] = None, rule_keywords: typing.Optional[str] = None) -> typing.Dict[str, typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.hpo.search_spaces.compose_search_space

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.compose_search_space
```
````

````{py:function} get_search_space(policy_name: str, raise_on_invalid: bool = True) -> typing.Dict[str, typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.hpo.search_spaces.get_search_space

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.get_search_space
```
````

````{py:function} _extract_params_from_config(config_class: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.hpo.search_spaces._extract_params_from_config

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces._extract_params_from_config
```
````

````{py:function} generate_policy_filters() -> None
:canonical: src.pipeline.simulations.hpo.search_spaces.generate_policy_filters

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_policy_filters
```
````

````{py:function} generate_route_improvement_interceptors() -> None
:canonical: src.pipeline.simulations.hpo.search_spaces.generate_route_improvement_interceptors

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_route_improvement_interceptors
```
````

````{py:function} generate_mandatory_selection_jobs() -> None
:canonical: src.pipeline.simulations.hpo.search_spaces.generate_mandatory_selection_jobs

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_mandatory_selection_jobs
```
````

````{py:function} generate_acceptance_criteria_rules() -> None
:canonical: src.pipeline.simulations.hpo.search_spaces.generate_acceptance_criteria_rules

```{autodoc2-docstring} src.pipeline.simulations.hpo.search_spaces.generate_acceptance_criteria_rules
```
````
