# {py:mod}`src.utils.logging.structured_logging`

```{py:module} src.utils.logging.structured_logging
```

```{autodoc2-docstring} src.utils.logging.structured_logging
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JsonFormatter <src.utils.logging.structured_logging.JsonFormatter>`
  - ```{autodoc2-docstring} src.utils.logging.structured_logging.JsonFormatter
    :summary:
    ```
* - {py:obj}`LogstashTcpHandler <src.utils.logging.structured_logging.LogstashTcpHandler>`
  - ```{autodoc2-docstring} src.utils.logging.structured_logging.LogstashTcpHandler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_structured_logger <src.utils.logging.structured_logging.get_structured_logger>`
  - ```{autodoc2-docstring} src.utils.logging.structured_logging.get_structured_logger
    :summary:
    ```
* - {py:obj}`log_test_metric <src.utils.logging.structured_logging.log_test_metric>`
  - ```{autodoc2-docstring} src.utils.logging.structured_logging.log_test_metric
    :summary:
    ```
* - {py:obj}`log_benchmark_metric <src.utils.logging.structured_logging.log_benchmark_metric>`
  - ```{autodoc2-docstring} src.utils.logging.structured_logging.log_benchmark_metric
    :summary:
    ```
````

### API

`````{py:class} JsonFormatter(fmt=None, datefmt=None, style='%', validate=True)
:canonical: src.utils.logging.structured_logging.JsonFormatter

Bases: {py:obj}`logging.Formatter`

```{autodoc2-docstring} src.utils.logging.structured_logging.JsonFormatter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.logging.structured_logging.JsonFormatter.__init__
```

````{py:method} format(record)
:canonical: src.utils.logging.structured_logging.JsonFormatter.format

```{autodoc2-docstring} src.utils.logging.structured_logging.JsonFormatter.format
```

````

`````

`````{py:class} LogstashTcpHandler(host: str = 'localhost', port: int = 5000)
:canonical: src.utils.logging.structured_logging.LogstashTcpHandler

Bases: {py:obj}`logging.Handler`

```{autodoc2-docstring} src.utils.logging.structured_logging.LogstashTcpHandler
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.logging.structured_logging.LogstashTcpHandler.__init__
```

````{py:method} emit(record)
:canonical: src.utils.logging.structured_logging.LogstashTcpHandler.emit

```{autodoc2-docstring} src.utils.logging.structured_logging.LogstashTcpHandler.emit
```

````

`````

````{py:function} get_structured_logger(name: str = 'wsmart.structured', level: int = logging.INFO, logstash_host: typing.Optional[str] = None, log_file: typing.Optional[str] = None) -> logging.Logger
:canonical: src.utils.logging.structured_logging.get_structured_logger

```{autodoc2-docstring} src.utils.logging.structured_logging.get_structured_logger
```
````

````{py:function} log_test_metric(name: str, value: typing.Any, logger_name: str = 'wsmart.structured')
:canonical: src.utils.logging.structured_logging.log_test_metric

```{autodoc2-docstring} src.utils.logging.structured_logging.log_test_metric
```
````

````{py:function} log_benchmark_metric(benchmark: str, metrics: typing.Dict[str, typing.Any], metadata: typing.Optional[typing.Dict[str, typing.Any]] = None, logger_name: str = 'wsmart.benchmark')
:canonical: src.utils.logging.structured_logging.log_benchmark_metric

```{autodoc2-docstring} src.utils.logging.structured_logging.log_benchmark_metric
```
````
