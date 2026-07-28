# {py:mod}`src.utils.decoding.decoding_utils`

```{py:module} src.utils.decoding.decoding_utils
```

```{autodoc2-docstring} src.utils.decoding.decoding_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CachedLookup <src.utils.decoding.decoding_utils.CachedLookup>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.CachedLookup
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`top_k_filter <src.utils.decoding.decoding_utils.top_k_filter>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.top_k_filter
    :summary:
    ```
* - {py:obj}`top_p_filter <src.utils.decoding.decoding_utils.top_p_filter>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.top_p_filter
    :summary:
    ```
* - {py:obj}`batchify <src.utils.decoding.decoding_utils.batchify>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.batchify
    :summary:
    ```
* - {py:obj}`unbatchify <src.utils.decoding.decoding_utils.unbatchify>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.unbatchify
    :summary:
    ```
* - {py:obj}`gather_by_index <src.utils.decoding.decoding_utils.gather_by_index>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.gather_by_index
    :summary:
    ```
* - {py:obj}`get_log_likelihood <src.utils.decoding.decoding_utils.get_log_likelihood>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.get_log_likelihood
    :summary:
    ```
* - {py:obj}`modify_logits_for_top_k_filtering <src.utils.decoding.decoding_utils.modify_logits_for_top_k_filtering>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.modify_logits_for_top_k_filtering
    :summary:
    ```
* - {py:obj}`modify_logits_for_top_p_filtering <src.utils.decoding.decoding_utils.modify_logits_for_top_p_filtering>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.modify_logits_for_top_p_filtering
    :summary:
    ```
* - {py:obj}`segment_topk_idx <src.utils.decoding.decoding_utils.segment_topk_idx>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.segment_topk_idx
    :summary:
    ```
* - {py:obj}`backtrack <src.utils.decoding.decoding_utils.backtrack>`
  - ```{autodoc2-docstring} src.utils.decoding.decoding_utils.backtrack
    :summary:
    ```
````

### API

````{py:function} top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.top_k_filter

```{autodoc2-docstring} src.utils.decoding.decoding_utils.top_k_filter
```
````

````{py:function} top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.top_p_filter

```{autodoc2-docstring} src.utils.decoding.decoding_utils.top_p_filter
```
````

````{py:function} batchify(td: tensordict.TensorDict, num_repeats: int) -> tensordict.TensorDict
:canonical: src.utils.decoding.decoding_utils.batchify

```{autodoc2-docstring} src.utils.decoding.decoding_utils.batchify
```
````

````{py:function} unbatchify(td: tensordict.TensorDict | torch.Tensor, num_repeats: int) -> tensordict.TensorDict | torch.Tensor
:canonical: src.utils.decoding.decoding_utils.unbatchify

```{autodoc2-docstring} src.utils.decoding.decoding_utils.unbatchify
```
````

````{py:function} gather_by_index(src: torch.Tensor, idx: torch.Tensor, dim: int = 1) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.gather_by_index

```{autodoc2-docstring} src.utils.decoding.decoding_utils.gather_by_index
```
````

````{py:function} get_log_likelihood(log_probs: torch.Tensor, actions: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, return_sum: bool = True) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.get_log_likelihood

```{autodoc2-docstring} src.utils.decoding.decoding_utils.get_log_likelihood
```
````

````{py:function} modify_logits_for_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.modify_logits_for_top_k_filtering

```{autodoc2-docstring} src.utils.decoding.decoding_utils.modify_logits_for_top_k_filtering
```
````

````{py:function} modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.modify_logits_for_top_p_filtering

```{autodoc2-docstring} src.utils.decoding.decoding_utils.modify_logits_for_top_p_filtering
```
````

````{py:function} segment_topk_idx(x: torch.Tensor, k: int, ids: torch.Tensor) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.segment_topk_idx

```{autodoc2-docstring} src.utils.decoding.decoding_utils.segment_topk_idx
```
````

````{py:function} backtrack(parents: list[torch.Tensor], actions: list[torch.Tensor]) -> torch.Tensor
:canonical: src.utils.decoding.decoding_utils.backtrack

```{autodoc2-docstring} src.utils.decoding.decoding_utils.backtrack
```
````

`````{py:class} CachedLookup(data=None, **kwargs)
:canonical: src.utils.decoding.decoding_utils.CachedLookup

```{autodoc2-docstring} src.utils.decoding.decoding_utils.CachedLookup
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.decoding_utils.CachedLookup.__init__
```

````{py:method} __getitem__(key)
:canonical: src.utils.decoding.decoding_utils.CachedLookup.__getitem__

```{autodoc2-docstring} src.utils.decoding.decoding_utils.CachedLookup.__getitem__
```

````

`````
