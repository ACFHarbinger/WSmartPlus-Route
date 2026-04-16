# {py:mod}`src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree`

```{py:module} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConstantNode <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode
    :summary:
    ```
* - {py:obj}`TerminalNode <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode
    :summary:
    ```
* - {py:obj}`FunctionNode <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`protected_div <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.protected_div>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.protected_div
    :summary:
    ```
* - {py:obj}`_collect_mutable_points <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._collect_mutable_points>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._collect_mutable_points
    :summary:
    ```
* - {py:obj}`_get_subtree <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._get_subtree>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._get_subtree
    :summary:
    ```
* - {py:obj}`_set_subtree <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._set_subtree>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._set_subtree
    :summary:
    ```
* - {py:obj}`_random_tree <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._random_tree>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._random_tree
    :summary:
    ```
* - {py:obj}`_subtree_crossover <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._subtree_crossover>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._subtree_crossover
    :summary:
    ```
* - {py:obj}`_mutate <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._mutate>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._mutate
    :summary:
    ```
* - {py:obj}`compile_tree <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.compile_tree>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.compile_tree
    :summary:
    ```
* - {py:obj}`to_callable <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.to_callable>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.to_callable
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPNode <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
    :summary:
    ```
* - {py:obj}`_TERMINALS <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._TERMINALS>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._TERMINALS
    :summary:
    ```
* - {py:obj}`_FUNCTIONS <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._FUNCTIONS>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._FUNCTIONS
    :summary:
    ```
* - {py:obj}`MutablePoint <src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.MutablePoint>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.MutablePoint
    :summary:
    ```
````

### API

````{py:function} protected_div(a: float, b: float) -> float
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.protected_div

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.protected_div
```
````

`````{py:class} ConstantNode(val: float)
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.__init__
```

````{py:attribute} __slots__
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.__slots__
:value: >
   ('val',)

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.__slots__
```

````

````{py:method} evaluate(ctx: typing.Dict[str, float]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.evaluate
```

````

````{py:method} copy() -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.copy

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.copy
```

````

````{py:method} size() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.size

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.size
```

````

````{py:method} depth() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.depth

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.depth
```

````

````{py:method} compile() -> str
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.compile

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.ConstantNode.compile
```

````

`````

`````{py:class} TerminalNode(feature: str)
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.__init__
```

````{py:attribute} __slots__
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.__slots__
:value: >
   ('feature',)

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.__slots__
```

````

````{py:method} evaluate(ctx: typing.Dict[str, float]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.evaluate
```

````

````{py:method} copy() -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.copy

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.copy
```

````

````{py:method} size() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.size

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.size
```

````

````{py:method} depth() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.depth

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.depth
```

````

````{py:method} compile() -> str
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.compile

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.TerminalNode.compile
```

````

`````

`````{py:class} FunctionNode(fn: str, left: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode, right: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode)
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.__init__
```

````{py:attribute} __slots__
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.__slots__
:value: >
   ('fn', 'left', 'right')

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.__slots__
```

````

````{py:method} evaluate(ctx: typing.Dict[str, float]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.evaluate
```

````

````{py:method} copy() -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.copy

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.copy
```

````

````{py:method} size() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.size

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.size
```

````

````{py:method} depth() -> int
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.depth

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.depth
```

````

````{py:method} compile() -> str
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.compile

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode.compile
```

````

`````

````{py:data} GPNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
```

````

````{py:data} _TERMINALS
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._TERMINALS
:value: >
   ['node_profit', 'distance_to_route', 'insertion_cost', 'remaining_capacity']

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._TERMINALS
```

````

````{py:data} _FUNCTIONS
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._FUNCTIONS
:value: >
   ['ADD', 'SUB', 'MUL', 'DIV', 'MAX', 'MIN']

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._FUNCTIONS
```

````

````{py:data} MutablePoint
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.MutablePoint
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.MutablePoint
```

````

````{py:function} _collect_mutable_points(tree: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode) -> typing.List[src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.MutablePoint]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._collect_mutable_points

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._collect_mutable_points
```
````

````{py:function} _get_subtree(parent: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode, side: str) -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._get_subtree

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._get_subtree
```
````

````{py:function} _set_subtree(parent: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.FunctionNode, side: str, subtree: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode) -> None
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._set_subtree

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._set_subtree
```
````

````{py:function} _random_tree(depth: int, rng: random.Random) -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._random_tree

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._random_tree
```
````

````{py:function} _subtree_crossover(t1: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode, t2: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode, rng: random.Random, max_depth: int) -> typing.Tuple[src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode, src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._subtree_crossover

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._subtree_crossover
```
````

````{py:function} _mutate(tree: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode, depth: int, rng: random.Random, max_depth: int, replacement_depth: typing.Optional[int] = None) -> src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._mutate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree._mutate
```
````

````{py:function} compile_tree(tree: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode) -> str
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.compile_tree

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.compile_tree
```
````

````{py:function} to_callable(tree: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.GPNode) -> typing.Callable
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.to_callable

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.tree.to_callable
```
````
