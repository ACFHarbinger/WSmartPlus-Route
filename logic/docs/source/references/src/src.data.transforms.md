# {py:mod}`src.data.transforms`

```{py:module} src.data.transforms
```

```{autodoc2-docstring} src.data.transforms
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StateAugmentation <src.data.transforms.StateAugmentation>`
  - ```{autodoc2-docstring} src.data.transforms.StateAugmentation
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`batchify <src.data.transforms.batchify>`
  - ```{autodoc2-docstring} src.data.transforms.batchify
    :summary:
    ```
* - {py:obj}`dihedral_8_augmentation <src.data.transforms.dihedral_8_augmentation>`
  - ```{autodoc2-docstring} src.data.transforms.dihedral_8_augmentation
    :summary:
    ```
* - {py:obj}`dihedral_8_augmentation_wrapper <src.data.transforms.dihedral_8_augmentation_wrapper>`
  - ```{autodoc2-docstring} src.data.transforms.dihedral_8_augmentation_wrapper
    :summary:
    ```
* - {py:obj}`symmetric_transform <src.data.transforms.symmetric_transform>`
  - ```{autodoc2-docstring} src.data.transforms.symmetric_transform
    :summary:
    ```
* - {py:obj}`symmetric_augmentation <src.data.transforms.symmetric_augmentation>`
  - ```{autodoc2-docstring} src.data.transforms.symmetric_augmentation
    :summary:
    ```
* - {py:obj}`min_max_normalize <src.data.transforms.min_max_normalize>`
  - ```{autodoc2-docstring} src.data.transforms.min_max_normalize
    :summary:
    ```
* - {py:obj}`get_augment_function <src.data.transforms.get_augment_function>`
  - ```{autodoc2-docstring} src.data.transforms.get_augment_function
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <src.data.transforms.log>`
  - ```{autodoc2-docstring} src.data.transforms.log
    :summary:
    ```
````

### API

````{py:data} log
:canonical: src.data.transforms.log
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.data.transforms.log
```

````

````{py:function} batchify(td: tensordict.TensorDict, num_samples: int) -> tensordict.TensorDict
:canonical: src.data.transforms.batchify

```{autodoc2-docstring} src.data.transforms.batchify
```
````

````{py:function} dihedral_8_augmentation(xy: torch.Tensor) -> torch.Tensor
:canonical: src.data.transforms.dihedral_8_augmentation

```{autodoc2-docstring} src.data.transforms.dihedral_8_augmentation
```
````

````{py:function} dihedral_8_augmentation_wrapper(xy: torch.Tensor, reduce: bool = True, *args, **kw) -> torch.Tensor
:canonical: src.data.transforms.dihedral_8_augmentation_wrapper

```{autodoc2-docstring} src.data.transforms.dihedral_8_augmentation_wrapper
```
````

````{py:function} symmetric_transform(x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor, offset: float = 0.5)
:canonical: src.data.transforms.symmetric_transform

```{autodoc2-docstring} src.data.transforms.symmetric_transform
```
````

````{py:function} symmetric_augmentation(xy: torch.Tensor, num_augment: int = 8, first_augment: bool = False)
:canonical: src.data.transforms.symmetric_augmentation

```{autodoc2-docstring} src.data.transforms.symmetric_augmentation
```
````

````{py:function} min_max_normalize(x)
:canonical: src.data.transforms.min_max_normalize

```{autodoc2-docstring} src.data.transforms.min_max_normalize
```
````

````{py:function} get_augment_function(augment_fn: typing.Union[str, typing.Callable])
:canonical: src.data.transforms.get_augment_function

```{autodoc2-docstring} src.data.transforms.get_augment_function
```
````

`````{py:class} StateAugmentation(num_augment: int = 8, augment_fn: typing.Union[str, typing.Callable] = 'symmetric', first_aug_identity: bool = True, normalize: bool = False, feats: typing.Optional[typing.List[str]] = None)
:canonical: src.data.transforms.StateAugmentation

```{autodoc2-docstring} src.data.transforms.StateAugmentation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.transforms.StateAugmentation.__init__
```

````{py:method} __call__(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.data.transforms.StateAugmentation.__call__

```{autodoc2-docstring} src.data.transforms.StateAugmentation.__call__
```

````

`````
