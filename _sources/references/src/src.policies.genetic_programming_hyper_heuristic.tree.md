# {py:mod}`src.policies.genetic_programming_hyper_heuristic.tree`

```{py:module} src.policies.genetic_programming_hyper_heuristic.tree
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TerminalNode <src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode
    :summary:
    ```
* - {py:obj}`FunctionNode <src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_random_tree <src.policies.genetic_programming_hyper_heuristic.tree._random_tree>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._random_tree
    :summary:
    ```
* - {py:obj}`_subtree_crossover <src.policies.genetic_programming_hyper_heuristic.tree._subtree_crossover>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._subtree_crossover
    :summary:
    ```
* - {py:obj}`_mutate <src.policies.genetic_programming_hyper_heuristic.tree._mutate>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._mutate
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPNode <src.policies.genetic_programming_hyper_heuristic.tree.GPNode>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.GPNode
    :summary:
    ```
* - {py:obj}`_TERMINALS <src.policies.genetic_programming_hyper_heuristic.tree._TERMINALS>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._TERMINALS
    :summary:
    ```
* - {py:obj}`_FUNCTIONS <src.policies.genetic_programming_hyper_heuristic.tree._FUNCTIONS>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._FUNCTIONS
    :summary:
    ```
````

### API

`````{py:class} TerminalNode(feature: str)
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode.__init__
```

````{py:method} evaluate(ctx: typing.Dict[str, float]) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode.evaluate

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode.evaluate
```

````

````{py:method} copy() -> src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode.copy

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.TerminalNode.copy
```

````

`````

`````{py:class} FunctionNode(fn: str, left: typing.Any, right: typing.Any, llh_true: int, llh_false: int)
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode.__init__
```

````{py:method} evaluate(ctx: typing.Dict[str, float]) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode.evaluate

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode.evaluate
```

````

````{py:method} copy() -> src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode.copy

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.FunctionNode.copy
```

````

`````

````{py:data} GPNode
:canonical: src.policies.genetic_programming_hyper_heuristic.tree.GPNode
:value: >
   None

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree.GPNode
```

````

````{py:data} _TERMINALS
:canonical: src.policies.genetic_programming_hyper_heuristic.tree._TERMINALS
:value: >
   ['avg_node_profit', 'load_factor', 'route_count', 'iter_progress']

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._TERMINALS
```

````

````{py:data} _FUNCTIONS
:canonical: src.policies.genetic_programming_hyper_heuristic.tree._FUNCTIONS
:value: >
   ['IF_GT', 'MAX_LLH']

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._FUNCTIONS
```

````

````{py:function} _random_tree(depth: int, n_llh: int, rng: random.Random) -> src.policies.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.genetic_programming_hyper_heuristic.tree._random_tree

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._random_tree
```
````

````{py:function} _subtree_crossover(t1: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, t2: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, rng: random.Random) -> typing.Tuple[src.policies.genetic_programming_hyper_heuristic.tree.GPNode, src.policies.genetic_programming_hyper_heuristic.tree.GPNode]
:canonical: src.policies.genetic_programming_hyper_heuristic.tree._subtree_crossover

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._subtree_crossover
```
````

````{py:function} _mutate(tree: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, depth: int, n_llh: int, rng: random.Random) -> src.policies.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.genetic_programming_hyper_heuristic.tree._mutate

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.tree._mutate
```
````
