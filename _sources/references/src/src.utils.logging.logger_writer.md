# {py:mod}`src.utils.logging.logger_writer`

```{py:module} src.utils.logging.logger_writer
```

```{autodoc2-docstring} src.utils.logging.logger_writer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LoggerWriter <src.utils.logging.logger_writer.LoggerWriter>`
  - ```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_logger_redirection <src.utils.logging.logger_writer.setup_logger_redirection>`
  - ```{autodoc2-docstring} src.utils.logging.logger_writer.setup_logger_redirection
    :summary:
    ```
````

### API

`````{py:class} LoggerWriter(terminal, filename, echo_to_terminal=True)
:canonical: src.utils.logging.logger_writer.LoggerWriter

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.__init__
```

````{py:method} write(message)
:canonical: src.utils.logging.logger_writer.LoggerWriter.write

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.write
```

````

````{py:method} flush()
:canonical: src.utils.logging.logger_writer.LoggerWriter.flush

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.flush
```

````

````{py:method} close()
:canonical: src.utils.logging.logger_writer.LoggerWriter.close

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.close
```

````

````{py:method} isatty()
:canonical: src.utils.logging.logger_writer.LoggerWriter.isatty

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.isatty
```

````

````{py:method} fileno()
:canonical: src.utils.logging.logger_writer.LoggerWriter.fileno

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.fileno
```

````

````{py:method} __getattr__(name)
:canonical: src.utils.logging.logger_writer.LoggerWriter.__getattr__

```{autodoc2-docstring} src.utils.logging.logger_writer.LoggerWriter.__getattr__
```

````

`````

````{py:function} setup_logger_redirection(log_file=None, silent=False)
:canonical: src.utils.logging.logger_writer.setup_logger_redirection

```{autodoc2-docstring} src.utils.logging.logger_writer.setup_logger_redirection
```
````
