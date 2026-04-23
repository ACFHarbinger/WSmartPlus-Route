# {py:mod}`src.data.distributions.statistical_bernoulli_gamma_mixture`

```{py:module} src.data.distributions.statistical_bernoulli_gamma_mixture
```

```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BernoulliGammaMixture <src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture
    :summary:
    ```
````

### API

`````{py:class} BernoulliGammaMixture(p: typing.Union[float, torch.Tensor] = 0.5, alpha: typing.Union[float, torch.Tensor] = 2.0, theta: typing.Union[float, torch.Tensor] = 2.0)
:canonical: src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_bernoulli_gamma_mixture.BernoulliGammaMixture._sample_array
```

````

`````
