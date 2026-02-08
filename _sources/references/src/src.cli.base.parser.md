# {py:mod}`src.cli.base.parser`

```{py:module} src.cli.base.parser
```

```{autodoc2-docstring} src.cli.base.parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConfigsParser <src.cli.base.parser.ConfigsParser>`
  - ```{autodoc2-docstring} src.cli.base.parser.ConfigsParser
    :summary:
    ```
````

### API

`````{py:class} ConfigsParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
:canonical: src.cli.base.parser.ConfigsParser

Bases: {py:obj}`argparse.ArgumentParser`

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser.__init__
```

````{py:method} _str_to_nargs(nargs: typing.Union[str, typing.Sequence]) -> typing.Union[str, typing.Sequence]
:canonical: src.cli.base.parser.ConfigsParser._str_to_nargs

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser._str_to_nargs
```

````

````{py:method} _process_args(namespace: argparse.Namespace) -> None
:canonical: src.cli.base.parser.ConfigsParser._process_args

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser._process_args
```

````

````{py:method} parse_command(args: typing.Optional[typing.Sequence[str]] = None) -> typing.Optional[str]
:canonical: src.cli.base.parser.ConfigsParser.parse_command

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser.parse_command
```

````

````{py:method} parse_process_args(args: typing.Optional[typing.List[str]] = None, command: typing.Optional[str] = None) -> typing.Tuple[typing.Optional[str], typing.Dict[str, typing.Any]]
:canonical: src.cli.base.parser.ConfigsParser.parse_process_args

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser.parse_process_args
```

````

````{py:method} error_message(message: str, print_help: bool = True) -> None
:canonical: src.cli.base.parser.ConfigsParser.error_message

```{autodoc2-docstring} src.cli.base.parser.ConfigsParser.error_message
```

````

`````
