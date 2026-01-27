# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.test.test_suite`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PyTestRunner <src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner
    :summary:
    ```
````

### API

`````{py:class} PyTestRunner(test_dir: str = 'tests')
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.__init__
```

````{py:method} _discover_test_modules() -> typing.List[str]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner._discover_test_modules

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner._discover_test_modules
```

````

````{py:method} _build_pytest_command(modules: typing.Optional[typing.List[str]] = None, test_class: typing.Optional[str] = None, test_method: typing.Optional[str] = None, verbose: bool = False, coverage: bool = False, markers: typing.Optional[str] = None, failed_first: bool = False, maxfail: typing.Optional[int] = None, capture: str = 'auto', tb_style: str = 'auto', parallel: bool = False, keyword: typing.Optional[str] = None) -> typing.List[str]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner._build_pytest_command

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner._build_pytest_command
```

````

````{py:method} run_tests(**kwargs) -> int
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.run_tests

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.run_tests
```

````

````{py:method} list_modules() -> None
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.list_modules

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.list_modules
```

````

````{py:method} list_tests(module: typing.Optional[str] = None) -> None
:canonical: src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.list_tests

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.test.test_suite.PyTestRunner.list_tests
```

````

`````
