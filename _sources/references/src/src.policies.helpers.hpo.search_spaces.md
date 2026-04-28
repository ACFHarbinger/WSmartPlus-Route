# {py:mod}`src.policies.helpers.hpo.search_spaces`

```{py:module} src.policies.helpers.hpo.search_spaces
```

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_all_search_spaces <src.policies.helpers.hpo.search_spaces.load_all_search_spaces>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.load_all_search_spaces
    :summary:
    ```
* - {py:obj}`get_search_space <src.policies.helpers.hpo.search_spaces.get_search_space>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.get_search_space
    :summary:
    ```
* - {py:obj}`_extract_params_from_config <src.policies.helpers.hpo.search_spaces._extract_params_from_config>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces._extract_params_from_config
    :summary:
    ```
* - {py:obj}`generate_policy_filters <src.policies.helpers.hpo.search_spaces.generate_policy_filters>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_policy_filters
    :summary:
    ```
* - {py:obj}`generate_route_improvement_interceptors <src.policies.helpers.hpo.search_spaces.generate_route_improvement_interceptors>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_route_improvement_interceptors
    :summary:
    ```
* - {py:obj}`generate_mandatory_selection_jobs <src.policies.helpers.hpo.search_spaces.generate_mandatory_selection_jobs>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_mandatory_selection_jobs
    :summary:
    ```
* - {py:obj}`generate_acceptance_criteria_rules <src.policies.helpers.hpo.search_spaces.generate_acceptance_criteria_rules>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_acceptance_criteria_rules
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILTERS_DIR <src.policies.helpers.hpo.search_spaces.FILTERS_DIR>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.FILTERS_DIR
    :summary:
    ```
* - {py:obj}`INTERCEPTORS_DIR <src.policies.helpers.hpo.search_spaces.INTERCEPTORS_DIR>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.INTERCEPTORS_DIR
    :summary:
    ```
* - {py:obj}`JOBS_DIR <src.policies.helpers.hpo.search_spaces.JOBS_DIR>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.JOBS_DIR
    :summary:
    ```
* - {py:obj}`RULES_DIR <src.policies.helpers.hpo.search_spaces.RULES_DIR>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.RULES_DIR
    :summary:
    ```
* - {py:obj}`POLICY_SEARCH_SPACES <src.policies.helpers.hpo.search_spaces.POLICY_SEARCH_SPACES>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.POLICY_SEARCH_SPACES
    :summary:
    ```
````

### API

````{py:data} FILTERS_DIR
:canonical: src.policies.helpers.hpo.search_spaces.FILTERS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.FILTERS_DIR
```

````

````{py:data} INTERCEPTORS_DIR
:canonical: src.policies.helpers.hpo.search_spaces.INTERCEPTORS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.INTERCEPTORS_DIR
```

````

````{py:data} JOBS_DIR
:canonical: src.policies.helpers.hpo.search_spaces.JOBS_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.JOBS_DIR
```

````

````{py:data} RULES_DIR
:canonical: src.policies.helpers.hpo.search_spaces.RULES_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.RULES_DIR
```

````

````{py:function} load_all_search_spaces() -> typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:canonical: src.policies.helpers.hpo.search_spaces.load_all_search_spaces

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.load_all_search_spaces
```
````

````{py:data} POLICY_SEARCH_SPACES
:canonical: src.policies.helpers.hpo.search_spaces.POLICY_SEARCH_SPACES
:type: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]
:value: >
   'load_all_search_spaces(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.POLICY_SEARCH_SPACES
```

````

````{py:function} get_search_space(policy_name: str) -> typing.Dict[str, typing.Dict[str, typing.Any]]
:canonical: src.policies.helpers.hpo.search_spaces.get_search_space

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.get_search_space
```
````

````{py:function} _extract_params_from_config(config_class: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.hpo.search_spaces._extract_params_from_config

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces._extract_params_from_config
```
````

````{py:function} generate_policy_filters() -> None
:canonical: src.policies.helpers.hpo.search_spaces.generate_policy_filters

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_policy_filters
```
````

````{py:function} generate_route_improvement_interceptors() -> None
:canonical: src.policies.helpers.hpo.search_spaces.generate_route_improvement_interceptors

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_route_improvement_interceptors
```
````

````{py:function} generate_mandatory_selection_jobs() -> None
:canonical: src.policies.helpers.hpo.search_spaces.generate_mandatory_selection_jobs

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_mandatory_selection_jobs
```
````

````{py:function} generate_acceptance_criteria_rules() -> None
:canonical: src.policies.helpers.hpo.search_spaces.generate_acceptance_criteria_rules

```{autodoc2-docstring} src.policies.helpers.hpo.search_spaces.generate_acceptance_criteria_rules
```
````
