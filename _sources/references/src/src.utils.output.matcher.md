# {py:mod}`src.utils.output.matcher`

```{py:module} src.utils.output.matcher
```

```{autodoc2-docstring} src.utils.output.matcher
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyFilter <src.utils.output.matcher.PolicyFilter>`
  - ```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parse_slug <src.utils.output.matcher._parse_slug>`
  - ```{autodoc2-docstring} src.utils.output.matcher._parse_slug
    :summary:
    ```
* - {py:obj}`slug_matches_filter <src.utils.output.matcher.slug_matches_filter>`
  - ```{autodoc2-docstring} src.utils.output.matcher.slug_matches_filter
    :summary:
    ```
* - {py:obj}`display_name_matches_filter <src.utils.output.matcher.display_name_matches_filter>`
  - ```{autodoc2-docstring} src.utils.output.matcher.display_name_matches_filter
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DISTRIBUTIONS <src.utils.output.matcher.DISTRIBUTIONS>`
  - ```{autodoc2-docstring} src.utils.output.matcher.DISTRIBUTIONS
    :summary:
    ```
* - {py:obj}`MS_STRATEGIES <src.utils.output.matcher.MS_STRATEGIES>`
  - ```{autodoc2-docstring} src.utils.output.matcher.MS_STRATEGIES
    :summary:
    ```
* - {py:obj}`IMPROVERS <src.utils.output.matcher.IMPROVERS>`
  - ```{autodoc2-docstring} src.utils.output.matcher.IMPROVERS
    :summary:
    ```
````

### API

````{py:data} DISTRIBUTIONS
:canonical: src.utils.output.matcher.DISTRIBUTIONS
:type: typing.List[str]
:value: >
   ['emp', 'gamma1', 'gamma2', 'gamma3', 'gamma4']

```{autodoc2-docstring} src.utils.output.matcher.DISTRIBUTIONS
```

````

````{py:data} MS_STRATEGIES
:canonical: src.utils.output.matcher.MS_STRATEGIES
:type: typing.List[str]
:value: >
   ['lookahead', 'last_minute', 'last_minute_cf70', 'last_minute_cf80', 'last_minute_cf90', 'regular', ...

```{autodoc2-docstring} src.utils.output.matcher.MS_STRATEGIES
```

````

````{py:data} IMPROVERS
:canonical: src.utils.output.matcher.IMPROVERS
:type: typing.List[str]
:value: >
   ['ftsp', 'fast_tsp', 'rls', 'rds', 'random_local_search', 'random_descent_search', 'two_opt', 'or_op...

```{autodoc2-docstring} src.utils.output.matcher.IMPROVERS
```

````

`````{py:class} PolicyFilter
:canonical: src.utils.output.matcher.PolicyFilter

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter
```

````{py:attribute} distributions
:canonical: src.utils.output.matcher.PolicyFilter.distributions
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter.distributions
```

````

````{py:attribute} constructors
:canonical: src.utils.output.matcher.PolicyFilter.constructors
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter.constructors
```

````

````{py:attribute} ms_strategies
:canonical: src.utils.output.matcher.PolicyFilter.ms_strategies
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter.ms_strategies
```

````

````{py:attribute} improvers
:canonical: src.utils.output.matcher.PolicyFilter.improvers
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter.improvers
```

````

````{py:attribute} exact_match
:canonical: src.utils.output.matcher.PolicyFilter.exact_match
:type: bool
:value: >
   False

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter.exact_match
```

````

````{py:method} _ok(accepted: typing.Sequence[str], value: str) -> bool
:canonical: src.utils.output.matcher.PolicyFilter._ok

```{autodoc2-docstring} src.utils.output.matcher.PolicyFilter._ok
```

````

`````

````{py:function} _parse_slug(slug: str) -> dict
:canonical: src.utils.output.matcher._parse_slug

```{autodoc2-docstring} src.utils.output.matcher._parse_slug
```
````

````{py:function} slug_matches_filter(slug: str, f: src.utils.output.matcher.PolicyFilter) -> bool
:canonical: src.utils.output.matcher.slug_matches_filter

```{autodoc2-docstring} src.utils.output.matcher.slug_matches_filter
```
````

````{py:function} display_name_matches_filter(display_name: str, f: src.utils.output.matcher.PolicyFilter) -> bool
:canonical: src.utils.output.matcher.display_name_matches_filter

```{autodoc2-docstring} src.utils.output.matcher.display_name_matches_filter
```
````
