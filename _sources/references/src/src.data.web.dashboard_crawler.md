# {py:mod}`src.data.web.dashboard_crawler`

```{py:module} src.data.web.dashboard_crawler
```

```{autodoc2-docstring} src.data.web.dashboard_crawler
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_load_html <src.data.web.dashboard_crawler._load_html>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._load_html
    :summary:
    ```
* - {py:obj}`_find_todos_os_locais_table <src.data.web.dashboard_crawler._find_todos_os_locais_table>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._find_todos_os_locais_table
    :summary:
    ```
* - {py:obj}`_parse_table <src.data.web.dashboard_crawler._parse_table>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._parse_table
    :summary:
    ```
* - {py:obj}`extract_dataframe <src.data.web.dashboard_crawler.extract_dataframe>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler.extract_dataframe
    :summary:
    ```
* - {py:obj}`to_csv <src.data.web.dashboard_crawler.to_csv>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler.to_csv
    :summary:
    ```
* - {py:obj}`to_excel <src.data.web.dashboard_crawler.to_excel>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler.to_excel
    :summary:
    ```
* - {py:obj}`to_simulation_data <src.data.web.dashboard_crawler.to_simulation_data>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler.to_simulation_data
    :summary:
    ```
* - {py:obj}`_build_parser <src.data.web.dashboard_crawler._build_parser>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._build_parser
    :summary:
    ```
* - {py:obj}`main <src.data.web.dashboard_crawler.main>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_EXCLUDED_COL <src.data.web.dashboard_crawler._EXCLUDED_COL>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._EXCLUDED_COL
    :summary:
    ```
* - {py:obj}`_COLUMN_RENAME <src.data.web.dashboard_crawler._COLUMN_RENAME>`
  - ```{autodoc2-docstring} src.data.web.dashboard_crawler._COLUMN_RENAME
    :summary:
    ```
````

### API

````{py:data} _EXCLUDED_COL
:canonical: src.data.web.dashboard_crawler._EXCLUDED_COL
:value: >
   'Viagem / Origem'

```{autodoc2-docstring} src.data.web.dashboard_crawler._EXCLUDED_COL
```

````

````{py:data} _COLUMN_RENAME
:canonical: src.data.web.dashboard_crawler._COLUMN_RENAME
:type: typing.Dict[str, str]
:value: >
   None

```{autodoc2-docstring} src.data.web.dashboard_crawler._COLUMN_RENAME
```

````

````{py:function} _load_html(source: str) -> str
:canonical: src.data.web.dashboard_crawler._load_html

```{autodoc2-docstring} src.data.web.dashboard_crawler._load_html
```
````

````{py:function} _find_todos_os_locais_table(soup: bs4.BeautifulSoup) -> typing.Optional[bs4.BeautifulSoup]
:canonical: src.data.web.dashboard_crawler._find_todos_os_locais_table

```{autodoc2-docstring} src.data.web.dashboard_crawler._find_todos_os_locais_table
```
````

````{py:function} _parse_table(table: bs4.BeautifulSoup) -> pandas.DataFrame
:canonical: src.data.web.dashboard_crawler._parse_table

```{autodoc2-docstring} src.data.web.dashboard_crawler._parse_table
```
````

````{py:function} extract_dataframe(source: str) -> pandas.DataFrame
:canonical: src.data.web.dashboard_crawler.extract_dataframe

```{autodoc2-docstring} src.data.web.dashboard_crawler.extract_dataframe
```
````

````{py:function} to_csv(source: str, output_path: str, sep: str = ',') -> str
:canonical: src.data.web.dashboard_crawler.to_csv

```{autodoc2-docstring} src.data.web.dashboard_crawler.to_csv
```
````

````{py:function} to_excel(source: str, output_path: str, sheet_name: str = 'Todos os Locais') -> str
:canonical: src.data.web.dashboard_crawler.to_excel

```{autodoc2-docstring} src.data.web.dashboard_crawler.to_excel
```
````

````{py:function} to_simulation_data(source: str, n_bins: typing.Optional[int] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.data.web.dashboard_crawler.to_simulation_data

```{autodoc2-docstring} src.data.web.dashboard_crawler.to_simulation_data
```
````

````{py:function} _build_parser() -> argparse.ArgumentParser
:canonical: src.data.web.dashboard_crawler._build_parser

```{autodoc2-docstring} src.data.web.dashboard_crawler._build_parser
```
````

````{py:function} main(argv: typing.Optional[typing.List[str]] = None) -> None
:canonical: src.data.web.dashboard_crawler.main

```{autodoc2-docstring} src.data.web.dashboard_crawler.main
```
````
