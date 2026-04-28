# {py:mod}`src.cli`

```{py:module} src.cli
```

```{autodoc2-docstring} src.cli
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.cli.base
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.cli.fs_parser
src.cli.gui_parser
src.cli.ts_parser
src.cli.benchmark_parser
src.cli.registry
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_params <src.cli.parse_params>`
  - ```{autodoc2-docstring} src.cli.parse_params
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.cli.__all__>`
  - ```{autodoc2-docstring} src.cli.__all__
    :summary:
    ```
````

### API

````{py:function} parse_params() -> typing.Tuple[typing.Union[str, typing.Tuple[str, str]], typing.Dict[str, typing.Any]]
:canonical: src.cli.parse_params

```{autodoc2-docstring} src.cli.parse_params
```
````

````{py:data} __all__
:canonical: src.cli.__all__
:value: >
   ['parse_params', 'add_files_args', 'add_gui_args', 'add_test_suite_args', 'ConfigsParser', 'Lowercas...

```{autodoc2-docstring} src.cli.__all__
```

````
