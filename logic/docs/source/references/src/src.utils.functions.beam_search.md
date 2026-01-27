# {py:mod}`src.utils.functions.beam_search`

```{py:module} src.utils.functions.beam_search
```

```{autodoc2-docstring} src.utils.functions.beam_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchBeam <src.utils.functions.beam_search.BatchBeam>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam
    :summary:
    ```
* - {py:obj}`CachedLookup <src.utils.functions.beam_search.CachedLookup>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.CachedLookup
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`beam_search <src.utils.functions.beam_search.beam_search>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.beam_search
    :summary:
    ```
* - {py:obj}`get_beam_search_results <src.utils.functions.beam_search.get_beam_search_results>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.get_beam_search_results
    :summary:
    ```
* - {py:obj}`_beam_search <src.utils.functions.beam_search._beam_search>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search._beam_search
    :summary:
    ```
* - {py:obj}`segment_topk_idx <src.utils.functions.beam_search.segment_topk_idx>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.segment_topk_idx
    :summary:
    ```
* - {py:obj}`backtrack <src.utils.functions.beam_search.backtrack>`
  - ```{autodoc2-docstring} src.utils.functions.beam_search.backtrack
    :summary:
    ```
````

### API

````{py:function} beam_search(*args, **kwargs)
:canonical: src.utils.functions.beam_search.beam_search

```{autodoc2-docstring} src.utils.functions.beam_search.beam_search
```
````

````{py:function} get_beam_search_results(beams, final_state)
:canonical: src.utils.functions.beam_search.get_beam_search_results

```{autodoc2-docstring} src.utils.functions.beam_search.get_beam_search_results
```
````

````{py:function} _beam_search(state, beam_size, propose_expansions=None, keep_states=False)
:canonical: src.utils.functions.beam_search._beam_search

```{autodoc2-docstring} src.utils.functions.beam_search._beam_search
```
````

`````{py:class} BatchBeam
:canonical: src.utils.functions.beam_search.BatchBeam

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam
```

````{py:attribute} score
:canonical: src.utils.functions.beam_search.BatchBeam.score
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.score
```

````

````{py:attribute} state
:canonical: src.utils.functions.beam_search.BatchBeam.state
:type: None
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.state
```

````

````{py:attribute} parent
:canonical: src.utils.functions.beam_search.BatchBeam.parent
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.parent
```

````

````{py:attribute} action
:canonical: src.utils.functions.beam_search.BatchBeam.action
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.action
```

````

````{py:attribute} batch_size
:canonical: src.utils.functions.beam_search.BatchBeam.batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.batch_size
```

````

````{py:attribute} device
:canonical: src.utils.functions.beam_search.BatchBeam.device
:type: None
:value: >
   None

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.device
```

````

````{py:property} ids
:canonical: src.utils.functions.beam_search.BatchBeam.ids

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.ids
```

````

````{py:method} __getitem__(key)
:canonical: src.utils.functions.beam_search.BatchBeam.__getitem__

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.__getitem__
```

````

````{py:method} initialize(state)
:canonical: src.utils.functions.beam_search.BatchBeam.initialize
:staticmethod:

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.initialize
```

````

````{py:method} propose_expansions()
:canonical: src.utils.functions.beam_search.BatchBeam.propose_expansions

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.propose_expansions
```

````

````{py:method} expand(parent, action, score=None)
:canonical: src.utils.functions.beam_search.BatchBeam.expand

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.expand
```

````

````{py:method} topk(k)
:canonical: src.utils.functions.beam_search.BatchBeam.topk

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.topk
```

````

````{py:method} all_finished()
:canonical: src.utils.functions.beam_search.BatchBeam.all_finished

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.all_finished
```

````

````{py:method} cpu()
:canonical: src.utils.functions.beam_search.BatchBeam.cpu

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.cpu
```

````

````{py:method} to(device)
:canonical: src.utils.functions.beam_search.BatchBeam.to

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.to
```

````

````{py:method} clear_state()
:canonical: src.utils.functions.beam_search.BatchBeam.clear_state

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.clear_state
```

````

````{py:method} size()
:canonical: src.utils.functions.beam_search.BatchBeam.size

```{autodoc2-docstring} src.utils.functions.beam_search.BatchBeam.size
```

````

`````

````{py:function} segment_topk_idx(x, k, ids)
:canonical: src.utils.functions.beam_search.segment_topk_idx

```{autodoc2-docstring} src.utils.functions.beam_search.segment_topk_idx
```
````

````{py:function} backtrack(parents, actions)
:canonical: src.utils.functions.beam_search.backtrack

```{autodoc2-docstring} src.utils.functions.beam_search.backtrack
```
````

`````{py:class} CachedLookup(data)
:canonical: src.utils.functions.beam_search.CachedLookup

Bases: {py:obj}`object`

```{autodoc2-docstring} src.utils.functions.beam_search.CachedLookup
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.beam_search.CachedLookup.__init__
```

````{py:method} __getitem__(key)
:canonical: src.utils.functions.beam_search.CachedLookup.__getitem__

```{autodoc2-docstring} src.utils.functions.beam_search.CachedLookup.__getitem__
```

````

`````
