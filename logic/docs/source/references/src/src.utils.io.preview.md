# {py:mod}`src.utils.io.preview`

```{py:module} src.utils.io.preview
```

```{autodoc2-docstring} src.utils.io.preview
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`preview_changes <src.utils.io.preview.preview_changes>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_changes
    :summary:
    ```
* - {py:obj}`preview_file_changes <src.utils.io.preview.preview_file_changes>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_file_changes
    :summary:
    ```
* - {py:obj}`preview_pattern_files_statistics <src.utils.io.preview.preview_pattern_files_statistics>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_pattern_files_statistics
    :summary:
    ```
* - {py:obj}`preview_file_statistics <src.utils.io.preview.preview_file_statistics>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_file_statistics
    :summary:
    ```
````

### API

````{py:function} preview_changes(root_directory, output_key='km', filename_pattern='log_*.json', process_func=None, update_val=0, input_keys=(None, None))
:canonical: src.utils.io.preview.preview_changes

```{autodoc2-docstring} src.utils.io.preview.preview_changes
```
````

````{py:function} preview_file_changes(file_path, output_key='km', process_func=None, update_val=0, input_keys=(None, None))
:canonical: src.utils.io.preview.preview_file_changes

```{autodoc2-docstring} src.utils.io.preview.preview_file_changes
```
````

````{py:function} preview_pattern_files_statistics(root_directory, filename_pattern='log_*.json', output_filename='output.json', output_key='km', process_func=None)
:canonical: src.utils.io.preview.preview_pattern_files_statistics

```{autodoc2-docstring} src.utils.io.preview.preview_pattern_files_statistics
```
````

````{py:function} preview_file_statistics(file_path, output_filename='output.json', output_key='km', process_func=None)
:canonical: src.utils.io.preview.preview_file_statistics

```{autodoc2-docstring} src.utils.io.preview.preview_file_statistics
```
````
