# {py:mod}`src.cli.tui`

```{py:module} src.cli.tui
```

```{autodoc2-docstring} src.cli.tui
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TerminalUI <src.cli.tui.TerminalUI>`
  - ```{autodoc2-docstring} src.cli.tui.TerminalUI
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`launch_tui <src.cli.tui.launch_tui>`
  - ```{autodoc2-docstring} src.cli.tui.launch_tui
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`tui_style <src.cli.tui.tui_style>`
  - ```{autodoc2-docstring} src.cli.tui.tui_style
    :summary:
    ```
````

### API

````{py:data} tui_style
:canonical: src.cli.tui.tui_style
:value: >
   'from_dict(...)'

```{autodoc2-docstring} src.cli.tui.tui_style
```

````

`````{py:class} TerminalUI()
:canonical: src.cli.tui.TerminalUI

```{autodoc2-docstring} src.cli.tui.TerminalUI
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.cli.tui.TerminalUI.__init__
```

````{py:method} display_header()
:canonical: src.cli.tui.TerminalUI.display_header

```{autodoc2-docstring} src.cli.tui.TerminalUI.display_header
```

````

````{py:method} _quick_select(title: str, text: str, values: typing.List[typing.Tuple[str, str]]) -> typing.Optional[str]
:canonical: src.cli.tui.TerminalUI._quick_select

```{autodoc2-docstring} src.cli.tui.TerminalUI._quick_select
```

````

````{py:method} _prompt(message: str, default: typing.Union[str, int, bool] = '', is_int: bool = False, is_bool: bool = False, choices: typing.Optional[typing.List[str]] = None) -> typing.Any
:canonical: src.cli.tui.TerminalUI._prompt

```{autodoc2-docstring} src.cli.tui.TerminalUI._prompt
```

````

````{py:method} select_subcommand() -> typing.Optional[str]
:canonical: src.cli.tui.TerminalUI.select_subcommand

```{autodoc2-docstring} src.cli.tui.TerminalUI.select_subcommand
```

````

````{py:method} configure_command(command: str) -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI.configure_command

```{autodoc2-docstring} src.cli.tui.TerminalUI.configure_command
```

````

````{py:method} _form_test_sim() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_test_sim

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_test_sim
```

````

````{py:method} _form_train() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_train

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_train
```

````

````{py:method} _form_mrl_train() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_mrl_train

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_mrl_train
```

````

````{py:method} _form_hp_optim() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_hp_optim

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_hp_optim
```

````

````{py:method} _form_eval() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_eval

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_eval
```

````

````{py:method} _form_gen_data() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_gen_data

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_gen_data
```

````

````{py:method} _form_file_system() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_file_system

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_file_system
```

````

````{py:method} _form_gui() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_gui

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_gui
```

````

````{py:method} _form_test_suite() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.cli.tui.TerminalUI._form_test_suite

```{autodoc2-docstring} src.cli.tui.TerminalUI._form_test_suite
```

````

````{py:method} run() -> typing.Optional[typing.Tuple[str, typing.Dict[str, typing.Any]]]
:canonical: src.cli.tui.TerminalUI.run

```{autodoc2-docstring} src.cli.tui.TerminalUI.run
```

````

`````

````{py:function} launch_tui() -> typing.Optional[typing.Tuple[str, typing.Dict[str, typing.Any]]]
:canonical: src.cli.tui.launch_tui

```{autodoc2-docstring} src.cli.tui.launch_tui
```
````
