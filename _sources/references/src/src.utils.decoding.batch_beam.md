# {py:mod}`src.utils.decoding.batch_beam`

```{py:module} src.utils.decoding.batch_beam
```

```{autodoc2-docstring} src.utils.decoding.batch_beam
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchBeam <src.utils.decoding.batch_beam.BatchBeam>`
  - ```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam
    :summary:
    ```
````

### API

`````{py:class} BatchBeam
:canonical: src.utils.decoding.batch_beam.BatchBeam

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam
```

````{py:attribute} score
:canonical: src.utils.decoding.batch_beam.BatchBeam.score
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.score
```

````

````{py:attribute} state
:canonical: src.utils.decoding.batch_beam.BatchBeam.state
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.state
```

````

````{py:attribute} parent
:canonical: src.utils.decoding.batch_beam.BatchBeam.parent
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.parent
```

````

````{py:attribute} action
:canonical: src.utils.decoding.batch_beam.BatchBeam.action
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.action
```

````

````{py:attribute} batch_size
:canonical: src.utils.decoding.batch_beam.BatchBeam.batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.batch_size
```

````

````{py:attribute} device
:canonical: src.utils.decoding.batch_beam.BatchBeam.device
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.device
```

````

````{py:property} ids
:canonical: src.utils.decoding.batch_beam.BatchBeam.ids

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.ids
```

````

````{py:method} __getitem__(key)
:canonical: src.utils.decoding.batch_beam.BatchBeam.__getitem__

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.__getitem__
```

````

````{py:method} initialize(state)
:canonical: src.utils.decoding.batch_beam.BatchBeam.initialize
:staticmethod:

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.initialize
```

````

````{py:method} propose_expansions()
:canonical: src.utils.decoding.batch_beam.BatchBeam.propose_expansions

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.propose_expansions
```

````

````{py:method} expand(parent, action, score=None)
:canonical: src.utils.decoding.batch_beam.BatchBeam.expand

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.expand
```

````

````{py:method} topk(k)
:canonical: src.utils.decoding.batch_beam.BatchBeam.topk

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.topk
```

````

````{py:method} all_finished()
:canonical: src.utils.decoding.batch_beam.BatchBeam.all_finished

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.all_finished
```

````

````{py:method} cpu()
:canonical: src.utils.decoding.batch_beam.BatchBeam.cpu

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.cpu
```

````

````{py:method} to(device)
:canonical: src.utils.decoding.batch_beam.BatchBeam.to

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.to
```

````

````{py:method} clear_state()
:canonical: src.utils.decoding.batch_beam.BatchBeam.clear_state

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.clear_state
```

````

````{py:method} size()
:canonical: src.utils.decoding.batch_beam.BatchBeam.size

```{autodoc2-docstring} src.utils.decoding.batch_beam.BatchBeam.size
```

````

`````
