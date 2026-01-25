# {py:mod}`src.utils.functions.decoding`

```{py:module} src.utils.functions.decoding
```

```{autodoc2-docstring} src.utils.functions.decoding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecodingStrategy <src.utils.functions.decoding.DecodingStrategy>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy
    :summary:
    ```
* - {py:obj}`Greedy <src.utils.functions.decoding.Greedy>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.Greedy
    :summary:
    ```
* - {py:obj}`Sampling <src.utils.functions.decoding.Sampling>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.Sampling
    :summary:
    ```
* - {py:obj}`BeamSearch <src.utils.functions.decoding.BeamSearch>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.BeamSearch
    :summary:
    ```
* - {py:obj}`Evaluate <src.utils.functions.decoding.Evaluate>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.Evaluate
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`top_k_filter <src.utils.functions.decoding.top_k_filter>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.top_k_filter
    :summary:
    ```
* - {py:obj}`top_p_filter <src.utils.functions.decoding.top_p_filter>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.top_p_filter
    :summary:
    ```
* - {py:obj}`batchify <src.utils.functions.decoding.batchify>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.batchify
    :summary:
    ```
* - {py:obj}`unbatchify <src.utils.functions.decoding.unbatchify>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.unbatchify
    :summary:
    ```
* - {py:obj}`gather_by_index <src.utils.functions.decoding.gather_by_index>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.gather_by_index
    :summary:
    ```
* - {py:obj}`get_decoding_strategy <src.utils.functions.decoding.get_decoding_strategy>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.get_decoding_strategy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DECODING_STRATEGY_REGISTRY <src.utils.functions.decoding.DECODING_STRATEGY_REGISTRY>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.DECODING_STRATEGY_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.utils.functions.decoding.__all__>`
  - ```{autodoc2-docstring} src.utils.functions.decoding.__all__
    :summary:
    ```
````

### API

`````{py:class} DecodingStrategy(temperature: float = 1.0, top_k: typing.Optional[int] = None, top_p: typing.Optional[float] = None, tanh_clipping: float = 0.0, mask_logits: bool = True, multistart: bool = False, num_starts: int = 1, select_best: bool = False, **kwargs)
:canonical: src.utils.functions.decoding.DecodingStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.decoding.DecodingStrategy.step
:abstractmethod:

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy.step
```

````

````{py:method} pre_decoder_hook(td: tensordict.TensorDict, env: any) -> typing.Tuple[tensordict.TensorDict, any, int]
:canonical: src.utils.functions.decoding.DecodingStrategy.pre_decoder_hook

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy.pre_decoder_hook
```

````

````{py:method} post_decoder_hook(td: tensordict.TensorDict, env: any, log_likelihood: torch.Tensor, actions: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, tensordict.TensorDict, any]
:canonical: src.utils.functions.decoding.DecodingStrategy.post_decoder_hook

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy.post_decoder_hook
```

````

````{py:method} _select_best(td: tensordict.TensorDict, log_likelihood: torch.Tensor, actions: torch.Tensor, num_starts: int) -> typing.Tuple[torch.Tensor, torch.Tensor, tensordict.TensorDict]
:canonical: src.utils.functions.decoding.DecodingStrategy._select_best

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy._select_best
```

````

````{py:method} _process_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
:canonical: src.utils.functions.decoding.DecodingStrategy._process_logits

```{autodoc2-docstring} src.utils.functions.decoding.DecodingStrategy._process_logits
```

````

`````

`````{py:class} Greedy(**kwargs)
:canonical: src.utils.functions.decoding.Greedy

Bases: {py:obj}`src.utils.functions.decoding.DecodingStrategy`

```{autodoc2-docstring} src.utils.functions.decoding.Greedy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.decoding.Greedy.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.decoding.Greedy.step

```{autodoc2-docstring} src.utils.functions.decoding.Greedy.step
```

````

`````

`````{py:class} Sampling(temperature: float = 1.0, top_k: typing.Optional[int] = None, top_p: typing.Optional[float] = None, tanh_clipping: float = 0.0, mask_logits: bool = True, multistart: bool = False, num_starts: int = 1, select_best: bool = False, **kwargs)
:canonical: src.utils.functions.decoding.Sampling

Bases: {py:obj}`src.utils.functions.decoding.DecodingStrategy`

```{autodoc2-docstring} src.utils.functions.decoding.Sampling
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.decoding.Sampling.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.decoding.Sampling.step

```{autodoc2-docstring} src.utils.functions.decoding.Sampling.step
```

````

`````

`````{py:class} BeamSearch(beam_width: int = 5, **kwargs)
:canonical: src.utils.functions.decoding.BeamSearch

Bases: {py:obj}`src.utils.functions.decoding.DecodingStrategy`

```{autodoc2-docstring} src.utils.functions.decoding.BeamSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.decoding.BeamSearch.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.decoding.BeamSearch.step

```{autodoc2-docstring} src.utils.functions.decoding.BeamSearch.step
```

````

`````

`````{py:class} Evaluate(actions: typing.Optional[torch.Tensor] = None, **kwargs)
:canonical: src.utils.functions.decoding.Evaluate

Bases: {py:obj}`src.utils.functions.decoding.DecodingStrategy`

```{autodoc2-docstring} src.utils.functions.decoding.Evaluate
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.functions.decoding.Evaluate.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.decoding.Evaluate.step

```{autodoc2-docstring} src.utils.functions.decoding.Evaluate.step
```

````

````{py:method} reset()
:canonical: src.utils.functions.decoding.Evaluate.reset

```{autodoc2-docstring} src.utils.functions.decoding.Evaluate.reset
```

````

`````

````{py:function} top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor
:canonical: src.utils.functions.decoding.top_k_filter

```{autodoc2-docstring} src.utils.functions.decoding.top_k_filter
```
````

````{py:function} top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor
:canonical: src.utils.functions.decoding.top_p_filter

```{autodoc2-docstring} src.utils.functions.decoding.top_p_filter
```
````

````{py:function} batchify(td: tensordict.TensorDict, num_repeats: int) -> tensordict.TensorDict
:canonical: src.utils.functions.decoding.batchify

```{autodoc2-docstring} src.utils.functions.decoding.batchify
```
````

````{py:function} unbatchify(td: tensordict.TensorDict, num_repeats: int) -> tensordict.TensorDict
:canonical: src.utils.functions.decoding.unbatchify

```{autodoc2-docstring} src.utils.functions.decoding.unbatchify
```
````

````{py:function} gather_by_index(src: torch.Tensor, idx: torch.Tensor, dim: int = 1) -> torch.Tensor
:canonical: src.utils.functions.decoding.gather_by_index

```{autodoc2-docstring} src.utils.functions.decoding.gather_by_index
```
````

````{py:data} DECODING_STRATEGY_REGISTRY
:canonical: src.utils.functions.decoding.DECODING_STRATEGY_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.utils.functions.decoding.DECODING_STRATEGY_REGISTRY
```

````

````{py:function} get_decoding_strategy(name: str, **kwargs) -> src.utils.functions.decoding.DecodingStrategy
:canonical: src.utils.functions.decoding.get_decoding_strategy

```{autodoc2-docstring} src.utils.functions.decoding.get_decoding_strategy
```
````

````{py:data} __all__
:canonical: src.utils.functions.decoding.__all__
:value: >
   ['DecodingStrategy', 'Greedy', 'Sampling', 'BeamSearch', 'Evaluate', 'get_decoding_strategy', 'top_k...

```{autodoc2-docstring} src.utils.functions.decoding.__all__
```

````
