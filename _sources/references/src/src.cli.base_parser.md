# {py:mod}`src.cli.base_parser`

```{py:module} src.cli.base_parser
```

```{autodoc2-docstring} src.cli.base_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConfigsParser <src.cli.base_parser.ConfigsParser>`
  - ```{autodoc2-docstring} src.cli.base_parser.ConfigsParser
    :summary:
    ```
* - {py:obj}`LowercaseAction <src.cli.base_parser.LowercaseAction>`
  - ```{autodoc2-docstring} src.cli.base_parser.LowercaseAction
    :summary:
    ```
* - {py:obj}`StoreDictKeyPair <src.cli.base_parser.StoreDictKeyPair>`
  - ```{autodoc2-docstring} src.cli.base_parser.StoreDictKeyPair
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`UpdateFunctionMapActionFactory <src.cli.base_parser.UpdateFunctionMapActionFactory>`
  - ```{autodoc2-docstring} src.cli.base_parser.UpdateFunctionMapActionFactory
    :summary:
    ```
````

### API

`````{py:class} ConfigsParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
:canonical: src.cli.base_parser.ConfigsParser

Bases: {py:obj}`argparse.ArgumentParser`

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser.__init__
```

````{py:method} _str_to_nargs(nargs: typing.Union[str, typing.Sequence]) -> typing.Union[str, typing.Sequence]
:canonical: src.cli.base_parser.ConfigsParser._str_to_nargs

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser._str_to_nargs
```

````

````{py:method} _process_args(namespace: argparse.Namespace) -> None
:canonical: src.cli.base_parser.ConfigsParser._process_args

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser._process_args
```

````

````{py:method} parse_command(args: typing.Optional[typing.Sequence[str]] = None) -> typing.Optional[str]
:canonical: src.cli.base_parser.ConfigsParser.parse_command

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser.parse_command
```

````

````{py:method} parse_process_args(args: typing.Optional[typing.List[str]] = None, command: typing.Optional[str] = None) -> typing.Tuple[typing.Optional[str], typing.Dict[str, typing.Any]]
:canonical: src.cli.base_parser.ConfigsParser.parse_process_args

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser.parse_process_args
```

````

````{py:method} error_message(message: str, print_help: bool = True) -> None
:canonical: src.cli.base_parser.ConfigsParser.error_message

```{autodoc2-docstring} src.cli.base_parser.ConfigsParser.error_message
```

````

`````

`````{py:class} LowercaseAction(option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None)
:canonical: src.cli.base_parser.LowercaseAction

Bases: {py:obj}`argparse.Action`

```{autodoc2-docstring} src.cli.base_parser.LowercaseAction
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.cli.base_parser.LowercaseAction.__init__
```

````{py:method} __call__(parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: typing.Any, option_string: typing.Optional[str] = None) -> None
:canonical: src.cli.base_parser.LowercaseAction.__call__

```{autodoc2-docstring} src.cli.base_parser.LowercaseAction.__call__
```

````

`````

`````{py:class} StoreDictKeyPair(option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None)
:canonical: src.cli.base_parser.StoreDictKeyPair

Bases: {py:obj}`argparse.Action`

```{autodoc2-docstring} src.cli.base_parser.StoreDictKeyPair
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.cli.base_parser.StoreDictKeyPair.__init__
```

````{py:method} __call__(parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: typing.List[str], option_string: typing.Optional[str] = None) -> None
:canonical: src.cli.base_parser.StoreDictKeyPair.__call__

```{autodoc2-docstring} src.cli.base_parser.StoreDictKeyPair.__call__
```

````

`````

````{py:function} UpdateFunctionMapActionFactory(inplace: bool = False) -> type
:canonical: src.cli.base_parser.UpdateFunctionMapActionFactory

```{autodoc2-docstring} src.cli.base_parser.UpdateFunctionMapActionFactory
```
````
