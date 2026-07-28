# {py:mod}`src.configs.policies.other.acceptance_criteria`

```{py:module} src.configs.policies.other.acceptance_criteria
```

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OnlyImprovingConfig <src.configs.policies.other.acceptance_criteria.OnlyImprovingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.OnlyImprovingConfig
    :summary:
    ```
* - {py:obj}`ImprovingAndEqualConfig <src.configs.policies.other.acceptance_criteria.ImprovingAndEqualConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ImprovingAndEqualConfig
    :summary:
    ```
* - {py:obj}`AllMovesAcceptedConfig <src.configs.policies.other.acceptance_criteria.AllMovesAcceptedConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AllMovesAcceptedConfig
    :summary:
    ```
* - {py:obj}`BoltzmannAcceptanceConfig <src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig
    :summary:
    ```
* - {py:obj}`DemonAlgorithmConfig <src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig
    :summary:
    ```
* - {py:obj}`GeneralizedTsallisSAConfig <src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig
    :summary:
    ```
* - {py:obj}`NonLinearGreatDelugeConfig <src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig
    :summary:
    ```
* - {py:obj}`EMCQConfig <src.configs.policies.other.acceptance_criteria.EMCQConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig
    :summary:
    ```
* - {py:obj}`AdaptiveBoltzmannConfig <src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig
    :summary:
    ```
* - {py:obj}`LateAcceptanceConfig <src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig
    :summary:
    ```
* - {py:obj}`StepCountingConfig <src.configs.policies.other.acceptance_criteria.StepCountingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.StepCountingConfig
    :summary:
    ```
* - {py:obj}`GreatDelugeConfig <src.configs.policies.other.acceptance_criteria.GreatDelugeConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GreatDelugeConfig
    :summary:
    ```
* - {py:obj}`ThresholdAcceptingConfig <src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig
    :summary:
    ```
* - {py:obj}`MonteCarloConfig <src.configs.policies.other.acceptance_criteria.MonteCarloConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.MonteCarloConfig
    :summary:
    ```
* - {py:obj}`AspirationConfig <src.configs.policies.other.acceptance_criteria.AspirationConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AspirationConfig
    :summary:
    ```
* - {py:obj}`AcceptanceConfig <src.configs.policies.other.acceptance_criteria.AcceptanceConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AcceptanceConfig
    :summary:
    ```
````

### API

````{py:class} OnlyImprovingConfig
:canonical: src.configs.policies.other.acceptance_criteria.OnlyImprovingConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.OnlyImprovingConfig
```

````

````{py:class} ImprovingAndEqualConfig
:canonical: src.configs.policies.other.acceptance_criteria.ImprovingAndEqualConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ImprovingAndEqualConfig
```

````

````{py:class} AllMovesAcceptedConfig
:canonical: src.configs.policies.other.acceptance_criteria.AllMovesAcceptedConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AllMovesAcceptedConfig
```

````

`````{py:class} BoltzmannAcceptanceConfig
:canonical: src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig
```

````{py:attribute} initial_temp
:canonical: src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.initial_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.initial_temp
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.alpha
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.BoltzmannAcceptanceConfig.seed
```

````

`````

`````{py:class} DemonAlgorithmConfig
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig
```

````{py:attribute} initial_credit
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.initial_credit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.initial_credit
```

````

````{py:attribute} is_stochastic
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.is_stochastic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.is_stochastic
```

````

````{py:attribute} max_demon_credit
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.max_demon_credit
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.max_demon_credit
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.maximization
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.DemonAlgorithmConfig.seed
```

````

`````

`````{py:class} GeneralizedTsallisSAConfig
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig
```

````{py:attribute} q
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.q
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.q
```

````

````{py:attribute} initial_temp
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.initial_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.initial_temp
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.alpha
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.seed
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GeneralizedTsallisSAConfig.maximization
```

````

`````

`````{py:class} NonLinearGreatDelugeConfig
:canonical: src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig
```

````{py:attribute} initial_level
:canonical: src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.initial_level
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.initial_level
```

````

````{py:attribute} beta
:canonical: src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.beta
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.beta
```

````

````{py:attribute} t_max
:canonical: src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.t_max
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.t_max
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.NonLinearGreatDelugeConfig.maximization
```

````

`````

`````{py:class} EMCQConfig
:canonical: src.configs.policies.other.acceptance_criteria.EMCQConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig
```

````{py:attribute} p
:canonical: src.configs.policies.other.acceptance_criteria.EMCQConfig.p
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig.p
```

````

````{py:attribute} p_boost
:canonical: src.configs.policies.other.acceptance_criteria.EMCQConfig.p_boost
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig.p_boost
```

````

````{py:attribute} q_threshold
:canonical: src.configs.policies.other.acceptance_criteria.EMCQConfig.q_threshold
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig.q_threshold
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.EMCQConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.EMCQConfig.seed
```

````

`````

`````{py:class} AdaptiveBoltzmannConfig
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig
```

````{py:attribute} p0
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.p0
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.p0
```

````

````{py:attribute} window_size
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.window_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.window_size
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.alpha
```

````

````{py:attribute} min_temp
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.min_temp
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.min_temp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.seed
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AdaptiveBoltzmannConfig.maximization
```

````

`````

`````{py:class} LateAcceptanceConfig
:canonical: src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig
```

````{py:attribute} history_length
:canonical: src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig.history_length
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig.history_length
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.LateAcceptanceConfig.maximization
```

````

`````

`````{py:class} StepCountingConfig
:canonical: src.configs.policies.other.acceptance_criteria.StepCountingConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.StepCountingConfig
```

````{py:attribute} step_limit
:canonical: src.configs.policies.other.acceptance_criteria.StepCountingConfig.step_limit
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.StepCountingConfig.step_limit
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.StepCountingConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.StepCountingConfig.maximization
```

````

`````

`````{py:class} GreatDelugeConfig
:canonical: src.configs.policies.other.acceptance_criteria.GreatDelugeConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GreatDelugeConfig
```

````{py:attribute} initial_level
:canonical: src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.initial_level
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.initial_level
```

````

````{py:attribute} decay_rate
:canonical: src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.decay_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.decay_rate
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.GreatDelugeConfig.maximization
```

````

`````

`````{py:class} ThresholdAcceptingConfig
:canonical: src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig
```

````{py:attribute} initial_threshold
:canonical: src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.initial_threshold
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.initial_threshold
```

````

````{py:attribute} decay_rate
:canonical: src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.decay_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.decay_rate
```

````

````{py:attribute} maximization
:canonical: src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.maximization
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.ThresholdAcceptingConfig.maximization
```

````

`````

`````{py:class} MonteCarloConfig
:canonical: src.configs.policies.other.acceptance_criteria.MonteCarloConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.MonteCarloConfig
```

````{py:attribute} p
:canonical: src.configs.policies.other.acceptance_criteria.MonteCarloConfig.p
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.MonteCarloConfig.p
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.acceptance_criteria.MonteCarloConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.MonteCarloConfig.seed
```

````

`````

````{py:class} AspirationConfig
:canonical: src.configs.policies.other.acceptance_criteria.AspirationConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AspirationConfig
```

````

`````{py:class} AcceptanceConfig
:canonical: src.configs.policies.other.acceptance_criteria.AcceptanceConfig

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AcceptanceConfig
```

````{py:attribute} method
:canonical: src.configs.policies.other.acceptance_criteria.AcceptanceConfig.method
:type: str
:value: >
   'oi'

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AcceptanceConfig.method
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.acceptance_criteria.AcceptanceConfig.params
:type: typing.Any
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.acceptance_criteria.AcceptanceConfig.params
```

````

`````
