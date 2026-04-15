# {py:mod}`src.policies.scenario_tree_extensive_form.tree`

```{py:module} src.policies.scenario_tree_extensive_form.tree
```

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScenarioNode <src.policies.scenario_tree_extensive_form.tree.ScenarioNode>`
  - ```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode
    :summary:
    ```
* - {py:obj}`ScenarioTree <src.policies.scenario_tree_extensive_form.tree.ScenarioTree>`
  - ```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree
    :summary:
    ```
````

### API

`````{py:class} ScenarioNode
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode
```

````{py:attribute} id
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.id
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.id
```

````

````{py:attribute} day
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.day
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.day
```

````

````{py:attribute} probability
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.probability
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.probability
```

````

````{py:attribute} realization
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.realization
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.realization
```

````

````{py:attribute} parent_id
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.parent_id
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.parent_id
```

````

````{py:attribute} children_ids
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioNode.children_ids
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioNode.children_ids
```

````

`````

`````{py:class} ScenarioTree(num_days: int, num_realizations: int, customers: typing.List[int], mean_increment: float, seed: typing.Optional[int] = 42)
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioTree

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree.__init__
```

````{py:method} _build_tree()
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioTree._build_tree

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree._build_tree
```

````

````{py:method} get_nodes_by_day(day: int) -> typing.List[src.policies.scenario_tree_extensive_form.tree.ScenarioNode]
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_nodes_by_day

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_nodes_by_day
```

````

````{py:method} get_root() -> src.policies.scenario_tree_extensive_form.tree.ScenarioNode
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_root

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_root
```

````

````{py:method} get_leaves() -> typing.List[src.policies.scenario_tree_extensive_form.tree.ScenarioNode]
:canonical: src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_leaves

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.tree.ScenarioTree.get_leaves
```

````

`````
