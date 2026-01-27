# {py:mod}`src.utils.logging.visualize_utils`

```{py:module} src.utils.logging.visualize_utils
```

```{autodoc2-docstring} src.utils.logging.visualize_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MyModelWrapper <src.utils.logging.visualize_utils.MyModelWrapper>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.MyModelWrapper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_batch <src.utils.logging.visualize_utils.get_batch>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.get_batch
    :summary:
    ```
* - {py:obj}`load_model_instance <src.utils.logging.visualize_utils.load_model_instance>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.load_model_instance
    :summary:
    ```
* - {py:obj}`plot_weight_trajectories <src.utils.logging.visualize_utils.plot_weight_trajectories>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_weight_trajectories
    :summary:
    ```
* - {py:obj}`log_weight_distributions <src.utils.logging.visualize_utils.log_weight_distributions>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.log_weight_distributions
    :summary:
    ```
* - {py:obj}`project_node_embeddings <src.utils.logging.visualize_utils.project_node_embeddings>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.project_node_embeddings
    :summary:
    ```
* - {py:obj}`plot_attention_heatmaps <src.utils.logging.visualize_utils.plot_attention_heatmaps>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_attention_heatmaps
    :summary:
    ```
* - {py:obj}`plot_logit_lens <src.utils.logging.visualize_utils.plot_logit_lens>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_logit_lens
    :summary:
    ```
* - {py:obj}`imitation_loss_fn <src.utils.logging.visualize_utils.imitation_loss_fn>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.imitation_loss_fn
    :summary:
    ```
* - {py:obj}`rl_loss_fn <src.utils.logging.visualize_utils.rl_loss_fn>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.rl_loss_fn
    :summary:
    ```
* - {py:obj}`plot_loss_landscape <src.utils.logging.visualize_utils.plot_loss_landscape>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_loss_landscape
    :summary:
    ```
* - {py:obj}`visualize_epoch <src.utils.logging.visualize_utils.visualize_epoch>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.visualize_epoch
    :summary:
    ```
* - {py:obj}`main <src.utils.logging.visualize_utils.main>`
  - ```{autodoc2-docstring} src.utils.logging.visualize_utils.main
    :summary:
    ```
````

### API

````{py:function} get_batch(device, size=50, batch_size=32, temporal_horizon=0)
:canonical: src.utils.logging.visualize_utils.get_batch

```{autodoc2-docstring} src.utils.logging.visualize_utils.get_batch
```
````

`````{py:class} MyModelWrapper(model)
:canonical: src.utils.logging.visualize_utils.MyModelWrapper

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.utils.logging.visualize_utils.MyModelWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.logging.visualize_utils.MyModelWrapper.__init__
```

````{py:method} forward(input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None)
:canonical: src.utils.logging.visualize_utils.MyModelWrapper.forward

```{autodoc2-docstring} src.utils.logging.visualize_utils.MyModelWrapper.forward
```

````

`````

````{py:function} load_model_instance(model_path, device, size=100, problem_name='wcvrp')
:canonical: src.utils.logging.visualize_utils.load_model_instance

```{autodoc2-docstring} src.utils.logging.visualize_utils.load_model_instance
```
````

````{py:function} plot_weight_trajectories(checkpoint_dir, output_file)
:canonical: src.utils.logging.visualize_utils.plot_weight_trajectories

```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_weight_trajectories
```
````

````{py:function} log_weight_distributions(model, epoch, log_dir, writer=None)
:canonical: src.utils.logging.visualize_utils.log_weight_distributions

```{autodoc2-docstring} src.utils.logging.visualize_utils.log_weight_distributions
```
````

````{py:function} project_node_embeddings(model, x_batch, log_dir, writer=None, epoch=0)
:canonical: src.utils.logging.visualize_utils.project_node_embeddings

```{autodoc2-docstring} src.utils.logging.visualize_utils.project_node_embeddings
```
````

````{py:function} plot_attention_heatmaps(model, output_dir, epoch=0)
:canonical: src.utils.logging.visualize_utils.plot_attention_heatmaps

```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_attention_heatmaps
```
````

````{py:function} plot_logit_lens(model, x_batch, output_file, epoch=0)
:canonical: src.utils.logging.visualize_utils.plot_logit_lens

```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_logit_lens
```
````

````{py:function} imitation_loss_fn(m, x_batch, pi_target, cost_weights=None)
:canonical: src.utils.logging.visualize_utils.imitation_loss_fn

```{autodoc2-docstring} src.utils.logging.visualize_utils.imitation_loss_fn
```
````

````{py:function} rl_loss_fn(m, x_batch, cost_weights=None)
:canonical: src.utils.logging.visualize_utils.rl_loss_fn

```{autodoc2-docstring} src.utils.logging.visualize_utils.rl_loss_fn
```
````

````{py:function} plot_loss_landscape(model, opts, output_dir, epoch=0, size=50, batch_size=16, resolution=10, span=1.0)
:canonical: src.utils.logging.visualize_utils.plot_loss_landscape

```{autodoc2-docstring} src.utils.logging.visualize_utils.plot_loss_landscape
```
````

````{py:function} visualize_epoch(model, problem, opts, epoch, tb_logger=None)
:canonical: src.utils.logging.visualize_utils.visualize_epoch

```{autodoc2-docstring} src.utils.logging.visualize_utils.visualize_epoch
```
````

````{py:function} main()
:canonical: src.utils.logging.visualize_utils.main

```{autodoc2-docstring} src.utils.logging.visualize_utils.main
```
````
