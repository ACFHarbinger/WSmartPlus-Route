# {py:mod}`src.utils.validation.trace_dependencies`

```{py:module} src.utils.validation.trace_dependencies
```

```{autodoc2-docstring} src.utils.validation.trace_dependencies
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ASTScopeVisitor <src.utils.validation.trace_dependencies.ASTScopeVisitor>`
  - ```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor
    :summary:
    ```
* - {py:obj}`DependencyGrapher <src.utils.validation.trace_dependencies.DependencyGrapher>`
  - ```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`main <src.utils.validation.trace_dependencies.main>`
  - ```{autodoc2-docstring} src.utils.validation.trace_dependencies.main
    :summary:
    ```
````

### API

`````{py:class} ASTScopeVisitor()
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor

Bases: {py:obj}`ast.NodeVisitor`

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.__init__
```

````{py:method} _add_node(name: str) -> typing.Optional[typing.Dict[str, typing.List[typing.Dict[str, typing.List[str]]]]]
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor._add_node

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor._add_node
```

````

````{py:method} visit_ClassDef(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_ClassDef

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_ClassDef
```

````

````{py:method} visit_FunctionDef(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_FunctionDef

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_FunctionDef
```

````

````{py:method} visit_AsyncFunctionDef(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_AsyncFunctionDef

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_AsyncFunctionDef
```

````

````{py:method} visit_Assign(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_Assign

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_Assign
```

````

````{py:method} visit_ImportFrom(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_ImportFrom

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_ImportFrom
```

````

````{py:method} visit_Import(node)
:canonical: src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_Import

```{autodoc2-docstring} src.utils.validation.trace_dependencies.ASTScopeVisitor.visit_Import
```

````

`````

`````{py:class} DependencyGrapher(project_root: str)
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher.__init__
```

````{py:method} _python_path_to_filepath(module_path: str) -> typing.Optional[str]
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._python_path_to_filepath

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._python_path_to_filepath
```

````

````{py:method} scan_project() -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher.scan_project

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher.scan_project
```

````

````{py:method} _parse_file(filepath: str) -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._parse_file

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._parse_file
```

````

````{py:method} trace_backward(current_file: str, target_name: str, visited: typing.Optional[typing.Set[typing.Tuple[str, str]]] = None) -> typing.Optional[str]
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher.trace_backward

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher.trace_backward
```

````

````{py:method} trace_forward(target_origin_file: str, target_original_name: str) -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher.trace_forward

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher.trace_forward
```

````

````{py:method} _build_node_uml(filepath: str) -> str
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._build_node_uml

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._build_node_uml
```

````

````{py:method} _build_edge_uml(source: str, target: str, direction: str) -> str
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._build_edge_uml

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._build_edge_uml
```

````

````{py:method} generate_graph(target_file: str, target_name: str) -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher.generate_graph

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher.generate_graph
```

````

````{py:method} _render_pyvis(target_node_id: str) -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._render_pyvis

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._render_pyvis
```

````

````{py:method} _inject_uml_panel(filepath: str) -> None
:canonical: src.utils.validation.trace_dependencies.DependencyGrapher._inject_uml_panel

```{autodoc2-docstring} src.utils.validation.trace_dependencies.DependencyGrapher._inject_uml_panel
```

````

`````

````{py:function} main() -> None
:canonical: src.utils.validation.trace_dependencies.main

```{autodoc2-docstring} src.utils.validation.trace_dependencies.main
```
````
