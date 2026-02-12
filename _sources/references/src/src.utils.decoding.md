# {py:mod}`src.utils.decoding`

```{py:module} src.utils.decoding
```

```{autodoc2-docstring} src.utils.decoding
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.utils.decoding.sampling
src.utils.decoding.base
src.utils.decoding.beam_search
src.utils.decoding.decoding_utils
src.utils.decoding.batch_beam
src.utils.decoding.evaluate
src.utils.decoding.greedy
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_decoding_strategy <src.utils.decoding.get_decoding_strategy>`
  - ```{autodoc2-docstring} src.utils.decoding.get_decoding_strategy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DECODING_STRATEGY_REGISTRY <src.utils.decoding.DECODING_STRATEGY_REGISTRY>`
  - ```{autodoc2-docstring} src.utils.decoding.DECODING_STRATEGY_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.utils.decoding.__all__>`
  - ```{autodoc2-docstring} src.utils.decoding.__all__
    :summary:
    ```
````

### API

````{py:data} DECODING_STRATEGY_REGISTRY
:canonical: src.utils.decoding.DECODING_STRATEGY_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.DECODING_STRATEGY_REGISTRY
```

````

````{py:function} get_decoding_strategy(name: str, **kwargs) -> src.utils.decoding.base.DecodingStrategy
:canonical: src.utils.decoding.get_decoding_strategy

```{autodoc2-docstring} src.utils.decoding.get_decoding_strategy
```
````

````{py:data} __all__
:canonical: src.utils.decoding.__all__
:value: >
   ['DecodingStrategy', 'Greedy', 'Sampling', 'BeamSearch', 'BatchBeam', 'beam_search', 'Evaluate', 'ge...

```{autodoc2-docstring} src.utils.decoding.__all__
```

````
