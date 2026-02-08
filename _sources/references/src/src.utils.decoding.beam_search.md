# {py:mod}`src.utils.decoding.beam_search`

```{py:module} src.utils.decoding.beam_search
```

```{autodoc2-docstring} src.utils.decoding.beam_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BeamSearch <src.utils.decoding.beam_search.BeamSearch>`
  - ```{autodoc2-docstring} src.utils.decoding.beam_search.BeamSearch
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`beam_search <src.utils.decoding.beam_search.beam_search>`
  - ```{autodoc2-docstring} src.utils.decoding.beam_search.beam_search
    :summary:
    ```
* - {py:obj}`get_beam_search_results <src.utils.decoding.beam_search.get_beam_search_results>`
  - ```{autodoc2-docstring} src.utils.decoding.beam_search.get_beam_search_results
    :summary:
    ```
* - {py:obj}`_beam_search <src.utils.decoding.beam_search._beam_search>`
  - ```{autodoc2-docstring} src.utils.decoding.beam_search._beam_search
    :summary:
    ```
````

### API

`````{py:class} BeamSearch(beam_width: int = 5, **kwargs)
:canonical: src.utils.decoding.beam_search.BeamSearch

Bases: {py:obj}`src.utils.decoding.base.DecodingStrategy`

```{autodoc2-docstring} src.utils.decoding.beam_search.BeamSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.beam_search.BeamSearch.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.decoding.beam_search.BeamSearch.step

```{autodoc2-docstring} src.utils.decoding.beam_search.BeamSearch.step
```

````

`````

````{py:function} beam_search(*args, **kwargs)
:canonical: src.utils.decoding.beam_search.beam_search

```{autodoc2-docstring} src.utils.decoding.beam_search.beam_search
```
````

````{py:function} get_beam_search_results(beams, final_state)
:canonical: src.utils.decoding.beam_search.get_beam_search_results

```{autodoc2-docstring} src.utils.decoding.beam_search.get_beam_search_results
```
````

````{py:function} _beam_search(state, beam_size, propose_expansions=None, keep_states=False)
:canonical: src.utils.decoding.beam_search._beam_search

```{autodoc2-docstring} src.utils.decoding.beam_search._beam_search
```
````
