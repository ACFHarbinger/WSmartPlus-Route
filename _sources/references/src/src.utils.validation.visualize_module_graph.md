# {py:mod}`src.utils.validation.visualize_module_graph`

```{py:module} src.utils.validation.visualize_module_graph
```

```{autodoc2-docstring} src.utils.validation.visualize_module_graph
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`file_to_module <src.utils.validation.visualize_module_graph.file_to_module>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.file_to_module
    :summary:
    ```
* - {py:obj}`collect_module_map <src.utils.validation.visualize_module_graph.collect_module_map>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.collect_module_map
    :summary:
    ```
* - {py:obj}`resolve_to_module <src.utils.validation.visualize_module_graph.resolve_to_module>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.resolve_to_module
    :summary:
    ```
* - {py:obj}`build_graph <src.utils.validation.visualize_module_graph.build_graph>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.build_graph
    :summary:
    ```
* - {py:obj}`get_layer <src.utils.validation.visualize_module_graph.get_layer>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.get_layer
    :summary:
    ```
* - {py:obj}`find_violations <src.utils.validation.visualize_module_graph.find_violations>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.find_violations
    :summary:
    ```
* - {py:obj}`condense_to_packages <src.utils.validation.visualize_module_graph.condense_to_packages>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.condense_to_packages
    :summary:
    ```
* - {py:obj}`generate_html <src.utils.validation.visualize_module_graph.generate_html>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.generate_html
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.visualize_module_graph.main>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.visualize_module_graph.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.SKIP_DIRS
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.visualize_module_graph.RED>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.RED
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.visualize_module_graph.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.YELLOW
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.visualize_module_graph.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.GREEN
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.validation.visualize_module_graph.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.CYAN
    :summary:
    ```
* - {py:obj}`BOLD <src.utils.validation.visualize_module_graph.BOLD>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.BOLD
    :summary:
    ```
* - {py:obj}`DIM <src.utils.validation.visualize_module_graph.DIM>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.DIM
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.visualize_module_graph.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.RESET
    :summary:
    ```
* - {py:obj}`DEFAULT_LAYERS <src.utils.validation.visualize_module_graph.DEFAULT_LAYERS>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.DEFAULT_LAYERS
    :summary:
    ```
* - {py:obj}`FORBIDDEN_DIRECTIONS <src.utils.validation.visualize_module_graph.FORBIDDEN_DIRECTIONS>`
  - ```{autodoc2-docstring} src.utils.validation.visualize_module_graph.FORBIDDEN_DIRECTIONS
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.visualize_module_graph.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.SKIP_DIRS
```

````

````{py:data} RED
:canonical: src.utils.validation.visualize_module_graph.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.RED
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.visualize_module_graph.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.YELLOW
```

````

````{py:data} GREEN
:canonical: src.utils.validation.visualize_module_graph.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.GREEN
```

````

````{py:data} CYAN
:canonical: src.utils.validation.visualize_module_graph.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.CYAN
```

````

````{py:data} BOLD
:canonical: src.utils.validation.visualize_module_graph.BOLD
:value: >
   '\x1b[1m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.BOLD
```

````

````{py:data} DIM
:canonical: src.utils.validation.visualize_module_graph.DIM
:value: >
   '\x1b[2m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.DIM
```

````

````{py:data} RESET
:canonical: src.utils.validation.visualize_module_graph.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.RESET
```

````

````{py:data} DEFAULT_LAYERS
:canonical: src.utils.validation.visualize_module_graph.DEFAULT_LAYERS
:type: typing.List[typing.Tuple[str, str, str]]
:value: >
   [('logic', 'Logic', '#3498db'), ('gui', 'GUI', '#9b59b6'), ('test', 'Tests', '#27ae60'), ('script', ...

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.DEFAULT_LAYERS
```

````

````{py:data} FORBIDDEN_DIRECTIONS
:canonical: src.utils.validation.visualize_module_graph.FORBIDDEN_DIRECTIONS
:type: typing.List[typing.Tuple[str, str]]
:value: >
   [('Logic', 'GUI')]

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.FORBIDDEN_DIRECTIONS
```

````

````{py:function} file_to_module(filepath: pathlib.Path, root: pathlib.Path) -> str
:canonical: src.utils.validation.visualize_module_graph.file_to_module

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.file_to_module
```
````

````{py:function} collect_module_map(root: pathlib.Path, exclude: typing.Set[str]) -> typing.Dict[str, pathlib.Path]
:canonical: src.utils.validation.visualize_module_graph.collect_module_map

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.collect_module_map
```
````

````{py:function} resolve_to_module(raw: str, level: int, current: str, known: typing.Set[str]) -> typing.Optional[str]
:canonical: src.utils.validation.visualize_module_graph.resolve_to_module

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.resolve_to_module
```
````

````{py:function} build_graph(root: pathlib.Path, exclude: typing.Set[str]) -> typing.Dict[str, typing.Set[str]]
:canonical: src.utils.validation.visualize_module_graph.build_graph

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.build_graph
```
````

````{py:function} get_layer(module: str, layers: typing.List[typing.Tuple[str, str, str]]) -> typing.Tuple[str, str]
:canonical: src.utils.validation.visualize_module_graph.get_layer

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.get_layer
```
````

````{py:function} find_violations(graph: typing.Dict[str, typing.Set[str]], layers: typing.List[typing.Tuple[str, str, str]], forbidden: typing.List[typing.Tuple[str, str]]) -> typing.List[typing.Tuple[str, str]]
:canonical: src.utils.validation.visualize_module_graph.find_violations

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.find_violations
```
````

````{py:function} condense_to_packages(graph: typing.Dict[str, typing.Set[str]], depth: int) -> typing.Tuple[typing.Dict[str, typing.Set[str]], typing.Dict[str, str]]
:canonical: src.utils.validation.visualize_module_graph.condense_to_packages

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.condense_to_packages
```
````

````{py:function} generate_html(graph: typing.Dict[str, typing.Set[str]], layers: typing.List[typing.Tuple[str, str, str]], violation_edges: typing.Set[typing.Tuple[str, str]], output: pathlib.Path, depth: int = 0) -> None
:canonical: src.utils.validation.visualize_module_graph.generate_html

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.generate_html
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.visualize_module_graph.main

```{autodoc2-docstring} src.utils.validation.visualize_module_graph.main
```
````
