# {py:mod}`src.policies.helpers.operators.sequence_merging.sequence_recombination`

```{py:module} src.policies.helpers.operators.sequence_merging.sequence_recombination
```

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sequence_single_point_crossover <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_single_point_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_single_point_crossover
    :summary:
    ```
* - {py:obj}`sequence_uniform_crossover <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_uniform_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_uniform_crossover
    :summary:
    ```
* - {py:obj}`sequence_order_preserving_crossover <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_order_preserving_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_order_preserving_crossover
    :summary:
    ```
* - {py:obj}`sequence_substitution_mutation <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_substitution_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_substitution_mutation
    :summary:
    ```
* - {py:obj}`sequence_insertion_mutation <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_insertion_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_insertion_mutation
    :summary:
    ```
* - {py:obj}`sequence_deletion_mutation <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_deletion_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_deletion_mutation
    :summary:
    ```
* - {py:obj}`sequence_transposition_mutation <src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_transposition_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_transposition_mutation
    :summary:
    ```
````

### API

````{py:function} sequence_single_point_crossover(parent1: typing.List[str], parent2: typing.List[str], rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[str], typing.List[str]]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_single_point_crossover

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_single_point_crossover
```
````

````{py:function} sequence_uniform_crossover(parent1: typing.List[str], parent2: typing.List[str], swap_prob: float = 0.5, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[str], typing.List[str]]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_uniform_crossover

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_uniform_crossover
```
````

````{py:function} sequence_order_preserving_crossover(parent1: typing.List[str], parent2: typing.List[str], rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_order_preserving_crossover

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_order_preserving_crossover
```
````

````{py:function} sequence_substitution_mutation(sequence: typing.List[str], op_pool: typing.List[str], n_mutations: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_substitution_mutation

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_substitution_mutation
```
````

````{py:function} sequence_insertion_mutation(sequence: typing.List[str], op_pool: typing.List[str], max_length: int = 20, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_insertion_mutation

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_insertion_mutation
```
````

````{py:function} sequence_deletion_mutation(sequence: typing.List[str], min_length: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_deletion_mutation

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_deletion_mutation
```
````

````{py:function} sequence_transposition_mutation(sequence: typing.List[str], rng: typing.Optional[random.Random] = None) -> typing.List[str]
:canonical: src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_transposition_mutation

```{autodoc2-docstring} src.policies.helpers.operators.sequence_merging.sequence_recombination.sequence_transposition_mutation
```
````
