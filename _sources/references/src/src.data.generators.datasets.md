# {py:mod}`src.data.generators.datasets`

```{py:module} src.data.generators.datasets
```

```{autodoc2-docstring} src.data.generators.datasets
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_datasets <src.data.generators.datasets.generate_datasets>`
  - ```{autodoc2-docstring} src.data.generators.datasets.generate_datasets
    :summary:
    ```
* - {py:obj}`_generate_problem_data <src.data.generators.datasets._generate_problem_data>`
  - ```{autodoc2-docstring} src.data.generators.datasets._generate_problem_data
    :summary:
    ```
* - {py:obj}`_process_instance_generation <src.data.generators.datasets._process_instance_generation>`
  - ```{autodoc2-docstring} src.data.generators.datasets._process_instance_generation
    :summary:
    ```
* - {py:obj}`_apply_noise_config <src.data.generators.datasets._apply_noise_config>`
  - ```{autodoc2-docstring} src.data.generators.datasets._apply_noise_config
    :summary:
    ```
* - {py:obj}`_generate_test_simulator_data <src.data.generators.datasets._generate_test_simulator_data>`
  - ```{autodoc2-docstring} src.data.generators.datasets._generate_test_simulator_data
    :summary:
    ```
* - {py:obj}`_generate_train_time_data <src.data.generators.datasets._generate_train_time_data>`
  - ```{autodoc2-docstring} src.data.generators.datasets._generate_train_time_data
    :summary:
    ```
* - {py:obj}`_generate_train_data <src.data.generators.datasets._generate_train_data>`
  - ```{autodoc2-docstring} src.data.generators.datasets._generate_train_data
    :summary:
    ```
* - {py:obj}`_verify_and_save <src.data.generators.datasets._verify_and_save>`
  - ```{autodoc2-docstring} src.data.generators.datasets._verify_and_save
    :summary:
    ```
````

### API

````{py:function} generate_datasets(opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets.generate_datasets

```{autodoc2-docstring} src.data.generators.datasets.generate_datasets
```
````

````{py:function} _generate_problem_data(problem: str, distributions: typing.Any, n_days: int, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._generate_problem_data

```{autodoc2-docstring} src.data.generators.datasets._generate_problem_data
```
````

````{py:function} _process_instance_generation(problem: str, dist: typing.Any, size: int, graph: typing.Any, n_days: int, datadir: str, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._process_instance_generation

```{autodoc2-docstring} src.data.generators.datasets._process_instance_generation
```
````

````{py:function} _apply_noise_config(builder: logic.src.data.builders.VRPInstanceBuilder, problem: str, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._apply_noise_config

```{autodoc2-docstring} src.data.generators.datasets._apply_noise_config
```
````

````{py:function} _generate_test_simulator_data(builder: logic.src.data.builders.VRPInstanceBuilder, n_days: int, datadir: str, dist: typing.Any, size: int, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._generate_test_simulator_data

```{autodoc2-docstring} src.data.generators.datasets._generate_test_simulator_data
```
````

````{py:function} _generate_train_time_data(builder: logic.src.data.builders.VRPInstanceBuilder, problem: str, n_days: int, datadir: str, dist: typing.Any, size: int, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._generate_train_time_data

```{autodoc2-docstring} src.data.generators.datasets._generate_train_time_data
```
````

````{py:function} _generate_train_data(builder: logic.src.data.builders.VRPInstanceBuilder, problem: str, datadir: str, dist: typing.Any, size: int, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.datasets._generate_train_data

```{autodoc2-docstring} src.data.generators.datasets._generate_train_data
```
````

````{py:function} _verify_and_save(builder: logic.src.data.builders.VRPInstanceBuilder, filename: str, opts: typing.Dict[str, typing.Any], is_td: bool = False) -> None
:canonical: src.data.generators.datasets._verify_and_save

```{autodoc2-docstring} src.data.generators.datasets._verify_and_save
```
````
