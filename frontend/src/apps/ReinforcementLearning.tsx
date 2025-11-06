// src/apps/reinforcement-learning.tsx
import React, { useState, useEffect } from 'react';
import { AppTab, AppContainer } from '../components/AppTab';
import {
  FormRow,
  SectionTitle,
  Input,
  NumberInput,
  Select,
  ToggleButton,
  CollapsibleSection,
} from '../components/UI';
import {
  PROBLEM_TYPES,
  DATA_DISTRIBUTIONS,
  DATA_DIST_PROBLEMS,
  BASELINES,
  MODELS,
  ENCODERS,
  NORMALIZATION_METHODS,
  ACTIVATION_FUNCTIONS,
  AGGREGATION_FUNCTIONS,
  LR_SCHEDULERS,
  OPTIMIZERS,
  WB_MODES,
  HOP_METHODS,
  HOP_METRICS,
  MRL_METHODS,
  CB_EXPLORATION_METHODS,
  RWA_MODELS,
  RWA_OPTIMIZERS,
  COUNTY_AREAS,
  WASTE_TYPES,
  DISTANCE_MATRIX_METHODS,
  VERTEX_METHODS,
  EDGE_METHODS,
} from '../constants';

export const RLTrainingTab: React.FC = () => {
  const [loadPath, setLoadPath] = useState('');
  const [resume, setResume] = useState(false);
  const [epochs, setEpochs] = useState(100);
  const [epochStart, setEpochStart] = useState(0);
  const [steps, setSteps] = useState(100);
  const [trainBatchSize, setTrainBatchSize] = useState(256);
  const [learningRateModel, setLearningRateModel] = useState(0.0001);
  const [learningRateCritic, setLearningRateCritic] = useState(0.001);
  const [gradNormClipping, setGradNormClipping] = useState(1.0);
  const [baseline, setBaseline] = useState(BASELINES[0]);
  const [exponentialAlpha, setExponentialAlpha] = useState(0.8);
  const [blWarmupEpochs, setBlWarmupEpochs] = useState(1);
  const [blDecayRate, setBlDecayRate] = useState(0.05);
  const [seed, setSeed] = useState(42);
  const [isVerbose, setIsVerbose] = useState(false);
  const [nWorkers, setNWorkers] = useState(navigator.hardwareConcurrency || 4);
  const [actorLrDecay, setActorLrDecay] = useState(1.0);
  const [criticLrDecay, setCriticLrDecay] = useState(1.0);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Training Parameters</SectionTitle>
      <FormRow label="Load Path:" htmlFor="rl_load_path">
        <Input id="rl_load_path" value={loadPath} onChange={e => setLoadPath(e.target.value)} placeholder="Path to pretrained model" />
      </FormRow>
      <FormRow label="Resume Training:">
        <ToggleButton checked={resume} onClick={() => setResume(c => !c)} variant="red-green">
          Resume from Checkpoint
        </ToggleButton>
      </FormRow>
      <FormRow label="Epochs:" htmlFor="rl_epochs">
        <NumberInput id="rl_epochs" value={epochs} onChange={e => setEpochs(Number(e.target.value))} min={1} max={10000} />
      </FormRow>
      <FormRow label="Start Epoch:" htmlFor="rl_epoch_start">
        <NumberInput id="rl_epoch_start" value={epochStart} onChange={e => setEpochStart(Number(e.target.value))} min={0} max={10000} />
      </FormRow>
      <FormRow label="Steps per Epoch:" htmlFor="rl_steps">
        <NumberInput id="rl_steps" value={steps} onChange={e => setSteps(Number(e.target.value))} min={1} max={10000} />
      </FormRow>
      <FormRow label="Train Batch Size:" htmlFor="rl_train_batch_size">
        <NumberInput id="rl_train_batch_size" value={trainBatchSize} onChange={e => setTrainBatchSize(Number(e.target.value))} min={1} max={1024} />
      </FormRow>
      <FormRow label="Actor Learning Rate:" htmlFor="rl_learning_rate_model">
        <NumberInput id="rl_learning_rate_model" value={learningRateModel} onChange={e => setLearningRateModel(Number(e.target.value))} min={0} max={0.1} step={0.0001} />
      </FormRow>
      <FormRow label="Critic Learning Rate:" htmlFor="rl_learning_rate_critic">
        <NumberInput id="rl_learning_rate_critic" value={learningRateCritic} onChange={e => setLearningRateCritic(Number(e.target.value))} min={0} max={0.1} step={0.0001} />
      </FormRow>
      <FormRow label="Gradient Norm Clipping:" htmlFor="rl_grad_norm_clipping">
        <NumberInput id="rl_grad_norm_clipping" value={gradNormClipping} onChange={e => setGradNormClipping(Number(e.target.value))} min={0} max={10} step={0.1} />
      </FormRow>
      <FormRow label="Baseline:" htmlFor="rl_baseline">
        <Select id="rl_baseline" value={baseline} onChange={e => setBaseline(e.target.value)}>
          {BASELINES.map(b => <option key={b} value={b}>{b.charAt(0).toUpperCase() + b.slice(1)}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Exponential Alpha:" htmlFor="rl_exponential_alpha">
        <NumberInput id="rl_exponential_alpha" value={exponentialAlpha} onChange={e => setExponentialAlpha(Number(e.target.value))} min={0} max={1} step={0.01} />
      </FormRow>
      <FormRow label="Baseline Warmup Epochs:" htmlFor="rl_bl_warmup_epochs">
        <NumberInput id="rl_bl_warmup_epochs" value={blWarmupEpochs} onChange={e => setBlWarmupEpochs(Number(e.target.value))} min={0} max={100} />
      </FormRow>
      <FormRow label="Baseline Decay Rate:" htmlFor="rl_bl_decay_rate">
        <NumberInput id="rl_bl_decay_rate" value={blDecayRate} onChange={e => setBlDecayRate(Number(e.target.value))} min={0} max={1} step={0.01} />
      </FormRow>
      <FormRow label="Random Seed:" htmlFor="rl_seed">
        <NumberInput id="rl_seed" value={seed} onChange={e => setSeed(Number(e.target.value))} min={0} max={100000} />
      </FormRow>
      <FormRow label="Verbose:">
        <ToggleButton checked={isVerbose} onClick={() => setIsVerbose(c => !c)} variant="red-green">
          Verbose Output
        </ToggleButton>
      </FormRow>
      <FormRow label="Number of Workers:" htmlFor="rl_n_workers">
        <NumberInput id="rl_n_workers" value={nWorkers} onChange={e => setNWorkers(Number(e.target.value))} min={1} max={navigator.hardwareConcurrency || 64} />
      </FormRow>
      <FormRow label="Actor LR Decay:" htmlFor="rl_actor_lr_decay">
        <NumberInput id="rl_actor_lr_decay" value={actorLrDecay} onChange={e => setActorLrDecay(Number(e.target.value))} min={0} max={1} step={0.01} />
      </FormRow>
      <FormRow label="Critic LR Decay:" htmlFor="rl_critic_lr_decay">
        <NumberInput id="rl_critic_lr_decay" value={criticLrDecay} onChange={e => setCriticLrDecay(Number(e.target.value))} min={0} max={1} step={0.01} />
      </FormRow>
    </AppTab>
  );
};

export const RLModelTab: React.FC = () => {
  const [model, setModel] = useState(MODELS['Attention Model']);
  const [encoder, setEncoder] = useState(ENCODERS['Graph Attention Encoder (GAT)']);
  const [embeddingDim, setEmbeddingDim] = useState(128);
  const [hiddenDim, setHiddenDim] = useState(128);
  const [nEncodeLayers, setNEncodeLayers] = useState(3);
  const [temporalHorizon, setTemporalHorizon] = useState(0);
  const [tanhClipping, setTanhClipping] = useState(10.0);
  const [normalization, setNormalization] = useState(NORMALIZATION_METHODS[0]);
  const [activation, setActivation] = useState(ACTIVATION_FUNCTIONS['ReLU']);
  const [dropout, setDropout] = useState(0.0);
  const [aggregationGraph, setAggregationGraph] = useState('');
  const [aggregation, setAggregation] = useState(AGGREGATION_FUNCTIONS['Mean']);
  const [nHeads, setNHeads] = useState(8);
  const [maskInner, setMaskInner] = useState(true);
  const [maskLogits, setMaskLogits] = useState(true);
  const [maskGraph, setMaskGraph] = useState(false);

  return (
    <AppTab>
      <FormRow label="Model:" htmlFor="rl_model">
        <Select id="rl_model" value={model} onChange={e => setModel(e.target.value)}>
          {Object.entries(MODELS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Encoder:" htmlFor="rl_encoder">
        <Select id="rl_encoder" value={encoder} onChange={e => setEncoder(e.target.value)}>
          {Object.entries(ENCODERS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Embedding Dimension:" htmlFor="rl_embedding_dim">
        <NumberInput id="rl_embedding_dim" value={embeddingDim} onChange={e => setEmbeddingDim(Number(e.target.value))} min={1} max={1024} />
      </FormRow>
      <FormRow label="Hidden Dimension:" htmlFor="rl_hidden_dim">
        <NumberInput id="rl_hidden_dim" value={hiddenDim} onChange={e => setHiddenDim(Number(e.target.value))} min={1} max={2048} />
      </FormRow>
      <FormRow label="Encode Layers:" htmlFor="rl_n_encode_layers">
        <NumberInput id="rl_n_encode_layers" value={nEncodeLayers} onChange={e => setNEncodeLayers(Number(e.target.value))} min={1} max={20} />
      </FormRow>
      <FormRow label="Temporal Horizon:" htmlFor="rl_temporal_horizon">
        <NumberInput id="rl_temporal_horizon" value={temporalHorizon} onChange={e => setTemporalHorizon(Number(e.target.value))} min={0} max={100} />
      </FormRow>
      <FormRow label="Tanh Clipping:" htmlFor="rl_tanh_clipping">
        <NumberInput id="rl_tanh_clipping" value={tanhClipping} onChange={e => setTanhClipping(Number(e.target.value))} min={0} max={100} step={0.1} />
      </FormRow>
      <FormRow label="Normalization:" htmlFor="rl_normalization">
        <Select id="rl_normalization" value={normalization} onChange={e => setNormalization(e.target.value)}>
          <option value="">Select Normalization</option>
          {NORMALIZATION_METHODS.map(m => <option key={m} value={m}>{m.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Activation:" htmlFor="rl_activation">
        <Select id="rl_activation" value={activation} onChange={e => setActivation(e.target.value)}>
          {Object.entries(ACTIVATION_FUNCTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Dropout:" htmlFor="rl_dropout">
        <NumberInput id="rl_dropout" value={dropout} onChange={e => setDropout(Number(e.target.value))} min={0} max={1} step={0.1} />
      </FormRow>
      <FormRow label="Graph Aggregation:" htmlFor="rl_aggregation_graph">
        <Select id="rl_aggregation_graph" value={aggregationGraph} onChange={e => setAggregationGraph(e.target.value)}>
          <option value="">Select Aggregation</option>
          {Object.entries(AGGREGATION_FUNCTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Node Aggregation:" htmlFor="rl_aggregation">
        <Select id="rl_aggregation" value={aggregation} onChange={e => setAggregation(e.target.value)}>
          {Object.entries(AGGREGATION_FUNCTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Attention Heads:" htmlFor="rl_n_heads">
        <NumberInput id="rl_n_heads" value={nHeads} onChange={e => setNHeads(Number(e.target.value))} min={1} max={32} />
      </FormRow>
      <FormRow label="Masking:">
        <div className="space-y-3">
          <ToggleButton checked={maskInner} onClick={() => setMaskInner(c => !c)} variant="green-red">
            Mask Inner
          </ToggleButton>
          <ToggleButton checked={maskLogits} onClick={() => setMaskLogits(c => !c)} variant="green-red">
            Mask Logits
          </ToggleButton>
          <ToggleButton checked={maskGraph} onClick={() => setMaskGraph(c => !c)} variant="red-green">
            Mask Graph
          </ToggleButton>
        </div>
      </FormRow>
    </AppTab>
  );
};

export const RLDataTab: React.FC = () => {
  const [problem, setProblem] = useState('TSP');
  const [dataDistribution, setDataDistribution] = useState(DATA_DISTRIBUTIONS['Gamma 1']);
  const [graphSize, setGraphSize] = useState(20);
  const [edgeThreshold, setEdgeThreshold] = useState(1.0);
  const [edgeMethod, setEdgeMethod] = useState(EDGE_METHODS['K-Nearest Neighbors (KNN)']);
  const [batchSize, setBatchSize] = useState(256);
  const [epochSize, setEpochSize] = useState(128000);
  const [valSize, setValSize] = useState(0);
  const [valDataset, setValDataset] = useState('');
  const [evalBatchSize, setEvalBatchSize] = useState(256);
  const [trainDataset, setTrainDataset] = useState('');
  const [area, setArea] = useState(COUNTY_AREAS['Rio Maior']);
  const [wasteType, setWasteType] = useState('plastic');
  const [distanceMethod, setDistanceMethod] = useState(DISTANCE_MATRIX_METHODS['Google Maps (GMaps)']);
  const [vertexMethod, setVertexMethod] = useState(VERTEX_METHODS['Min-Max Normalization']);

  const [isDataDistVisible, setIsDataDistVisible] = useState(false);

  useEffect(() => {
    setIsDataDistVisible(DATA_DIST_PROBLEMS.includes(problem.toUpperCase()));
  }, [problem]);

  return (
    <AppTab>
      <FormRow label="Problem:" htmlFor="rl_problem">
        <Select id="rl_problem" value={problem} onChange={e => setProblem(e.target.value)}>
          {PROBLEM_TYPES.map(p => <option key={p} value={p}>{p}</option>)}
        </Select>
      </FormRow>
      
      <FormRow label="Data Distribution:" htmlFor="rl_data_distribution" className={isDataDistVisible ? 'block' : 'hidden'}>
        <Select id="rl_data_distribution" value={dataDistribution} onChange={e => setDataDistribution(e.target.value)}>
          {Object.entries(DATA_DISTRIBUTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>

      <FormRow label="Graph Size:" htmlFor="rl_graph_size">
        <NumberInput id="rl_graph_size" value={graphSize} onChange={e => setGraphSize(Number(e.target.value))} min={1} max={1000} />
      </FormRow>
      <FormRow label="Edge Threshold (0.0 to 1.0):" htmlFor="rl_edge_threshold">
        <NumberInput id="rl_edge_threshold" value={edgeThreshold} onChange={e => setEdgeThreshold(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Edge Method:" htmlFor="rl_edge_method">
        <Select id="rl_edge_method" value={edgeMethod} onChange={e => setEdgeMethod(e.target.value)}>
          {Object.entries(EDGE_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Batch Size:" htmlFor="rl_batch_size">
        <NumberInput id="rl_batch_size" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} min={1} max={10000} />
      </FormRow>
      <FormRow label="Epoch Size:" htmlFor="rl_epoch_size">
        <NumberInput id="rl_epoch_size" value={epochSize} onChange={e => setEpochSize(Number(e.target.value))} min={1} max={1000000} />
      </FormRow>
      <FormRow label="Validation Size:" htmlFor="rl_val_size">
        <NumberInput id="rl_val_size" value={valSize} onChange={e => setValSize(Number(e.target.value))} min={0} max={100000} />
      </FormRow>
      <FormRow label="Validation Dataset:" htmlFor="rl_val_dataset">
        <Input id="rl_val_dataset" value={valDataset} onChange={e => setValDataset(e.target.value)} placeholder="path/to/val_dataset.pkl" />
      </FormRow>
      <FormRow label="Eval Batch Size:" htmlFor="rl_eval_batch_size">
        <NumberInput id="rl_eval_batch_size" value={evalBatchSize} onChange={e => setEvalBatchSize(Number(e.target.value))} min={1} max={10000} />
      </FormRow>
      <FormRow label="Train Dataset:" htmlFor="rl_train_dataset">
        <Input id="rl_train_dataset" value={trainDataset} onChange={e => setTrainDataset(e.target.value)} placeholder="path/to/train_dataset.pkl" />
      </FormRow>
      <FormRow label="County Area:" htmlFor="rl_area">
        <Select id="rl_area" value={area} onChange={e => setArea(e.target.value)}>
          {Object.entries(COUNTY_AREAS).filter(([k]) => k === 'Rio Maior' || k === 'Lisbon' || k === 'Porto').map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Waste Type:" htmlFor="rl_waste_type">
        <Select id="rl_waste_type" value={wasteType} onChange={e => setWasteType(e.target.value)}>
          {WASTE_TYPES.map(w => <option key={w} value={w.toLowerCase()}>{w}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Distance Method:" htmlFor="rl_distance_method">
        <Select id="rl_distance_method" value={distanceMethod} onChange={e => setDistanceMethod(e.target.value)}>
          {Object.entries(DISTANCE_MATRIX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Vertex Method:" htmlFor="rl_vertex_method">
        <Select id="rl_vertex_method" value={vertexMethod} onChange={e => setVertexMethod(e.target.value)}>
          {Object.entries(VERTEX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
    </AppTab>
  );
};

export const RLOptimizerTab: React.FC = () => {
  const [optimizer, setOptimizer] = useState(OPTIMIZERS['Root Mean Square Propagation (RMSProp)']);
  const [lrScheduler, setLrScheduler] = useState(LR_SCHEDULERS['Lambda Learning Rate']);
  const [lrDecay, setLrDecay] = useState(1.0);
  const [lrMinValue, setLrMinValue] = useState(0.0);

  return (
    <AppTab>
      <FormRow label="Optimizer:" htmlFor="rl_optimizer">
        <Select id="rl_optimizer" value={optimizer} onChange={e => setOptimizer(e.target.value)}>
          {Object.entries(OPTIMIZERS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Learning Rate Scheduler:" htmlFor="rl_lr_scheduler">
        <Select id="rl_lr_scheduler" value={lrScheduler} onChange={e => setLrScheduler(e.target.value)}>
          {Object.entries(LR_SCHEDULERS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Learning Rate Decay:" htmlFor="rl_lr_decay">
        <NumberInput id="rl_lr_decay" value={lrDecay} onChange={e => setLrDecay(Number(e.target.value))} min={0} max={2} step={0.1} />
      </FormRow>
      <FormRow label="Learning Rate Minimum Value:" htmlFor="rl_lr_min_value">
        <NumberInput id="rl_lr_min_value" value={lrMinValue} onChange={e => setLrMinValue(Number(e.target.value))} min={0} max={1} step={0.000001} />
      </FormRow>
    </AppTab>
  );
};

export const RLCostsTab: React.FC = () => {
  const [wWaste, setWWaste] = useState(0);
  const [wLength, setWLength] = useState(0);
  const [wOverflows, setWOverflows] = useState(0);
  const [wLost, setWLost] = useState(0);
  const [wPenalty, setWPenalty] = useState(0);
  const [wPrize, setWPrize] = useState(0);

  const CostInput: React.FC<{
    value: number,
    onChange: (val: number) => void,
    id: string
  }> = ({ value, onChange, id }) => (
    <NumberInput
      id={id}
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      min={-10}
      max={10}
      step={0.1}
      placeholder={value === 0 ? "Not set (0.0)" : ""}
    />
  );

  return (
    <AppTab>
      <FormRow label="Waste Weight:" htmlFor="rl_w_waste">
        <CostInput id="rl_w_waste" value={wWaste} onChange={setWWaste} />
      </FormRow>
      <FormRow label="Length Weight:" htmlFor="rl_w_length">
        <CostInput id="rl_w_length" value={wLength} onChange={setWLength} />
      </FormRow>
      <FormRow label="Overflows Weight:" htmlFor="rl_w_overflows">
        <CostInput id="rl_w_overflows" value={wOverflows} onChange={setWOverflows} />
      </FormRow>
      <FormRow label="Lost Weight:" htmlFor="rl_w_lost">
        <CostInput id="rl_w_lost" value={wLost} onChange={setWLost} />
      </FormRow>
      <FormRow label="Penalty Weight:" htmlFor="rl_w_penalty">
        <CostInput id="rl_w_penalty" value={wPenalty} onChange={setWPenalty} />
      </FormRow>
      <FormRow label="Prize Weight:" htmlFor="rl_w_prize">
        <CostInput id="rl_w_prize" value={wPrize} onChange={setWPrize} />
      </FormRow>
    </AppTab>
  );
};

export const RLOutputTab: React.FC = () => {
  const [logStep, setLogStep] = useState(50);
  const [logDir, setLogDir] = useState('logs');
  const [runName, setRunName] = useState('');
  const [outputDir, setOutputDir] = useState('model_weights');
  const [checkpointEpochs, setCheckpointEpochs] = useState(1);
  const [wandbMode, setWandbMode] = useState('');
  const [useTensorboard, setUseTensorboard] = useState(true); // 'no_tensorboard' is False
  const [useProgressBar, setUseProgressBar] = useState(true); // 'no_progress_bar' is False

  return (
    <AppTab>
      <FormRow label="Log Step:" htmlFor="rl_log_step">
        <NumberInput id="rl_log_step" value={logStep} onChange={e => setLogStep(Number(e.target.value))} min={1} max={10000} />
      </FormRow>
      <FormRow label="Log Directory:" htmlFor="rl_log_dir">
        <Input id="rl_log_dir" value={logDir} onChange={e => setLogDir(e.target.value)} />
      </FormRow>
      <FormRow label="Run Name:" htmlFor="rl_run_name">
        <Input id="rl_run_name" value={runName} onChange={e => setRunName(e.target.value)} placeholder="e.g., rl_model_run_1" />
      </FormRow>
      <FormRow label="Output Directory:" htmlFor="rl_output_dir">
        <Input id="rl_output_dir" value={outputDir} onChange={e => setOutputDir(e.target.value)} />
      </FormRow>
      <FormRow label="Checkpoint Epochs:" htmlFor="rl_checkpoint_epochs">
        <NumberInput id="rl_checkpoint_epochs" value={checkpointEpochs} onChange={e => setCheckpointEpochs(Number(e.target.value))} min={0} max={100} />
      </FormRow>
      <FormRow label="Weight and Biases Mode:" htmlFor="rl_wandb_mode">
        <Select id="rl_wandb_mode" value={wandbMode} onChange={e => setWandbMode(e.target.value)}>
          <option value="">Select Mode</option>
          {WB_MODES.map(m => <option key={m} value={m}>{m}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Options:">
        <div className="space-y-3">
          <ToggleButton checked={useTensorboard} onClick={() => setUseTensorboard(c => !c)} variant="green-red">
            TensorBoard Logger
          </ToggleButton>
          <ToggleButton checked={useProgressBar} onClick={() => setUseProgressBar(c => !c)} variant="green-red">
            Progress Bar
          </ToggleButton>
        </div>
      </FormRow>
    </AppTab>
  );
};

export const RLHPOTab: React.FC = () => {
  const [hopMethod, setHopMethod] = useState('');
  const [hopRange, setHopRange] = useState('0.0 2.0');
  const [hopEpochs, setHopEpochs] = useState(7);
  const [metric, setMetric] = useState(HOP_METRICS['Validation Loss']);
  const [cpuCores, setCpuCores] = useState(1);
  const [verbose, setVerbose] = useState(2);
  const [trainBest, setTrainBest] = useState(true);
  const [localMode, setLocalMode] = useState(false);
  const [numSamples, setNumSamples] = useState(20);
  const [nTrials, setNTrials] = useState(20);
  const [timeout, setTimeout] = useState('');
  const [nStartupTrials, setNStartupTrials] = useState(5);
  const [nWarmupSteps, setNWarmupSteps] = useState(3);
  const [intervalSteps, setIntervalSteps] = useState(1);
  const [eta, setEta] = useState(10.0);
  const [indpb, setIndpb] = useState(0.2);
  const [tournsize, setTournsize] = useState(3);
  const [cxpb, setCxpb] = useState(0.7);
  const [mutpb, setMutpb] = useState(0.2);
  const [nPop, setNPop] = useState(20);
  const [nGen, setNGen] = useState(10);
  const [maxTres, setMaxTres] = useState(14);
  const [reductionFactor, setReductionFactor] = useState(3);
  const [grid, setGrid] = useState('0.0 0.5 1.0 1.5 2.0');
  const [maxConc, setMaxConc] = useState(4);
  const [fevals, setFevals] = useState(100);
  const [maxFailures, setMaxFailures] = useState(3);

  return (
    <AppTab>
      <SectionTitle className="mt-0">General Settings</SectionTitle>
      <FormRow label="Optimization Method:" htmlFor="rl_hop_method">
        <Select id="rl_hop_method" value={hopMethod} onChange={e => setHopMethod(e.target.value)}>
          <option value="">Select Method</option>
          {Object.entries(HOP_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Hyper-Parameter Range:" htmlFor="rl_hop_range">
        <Input id="rl_hop_range" value={hopRange} onChange={e => setHopRange(e.target.value)} placeholder="Min-Max values (space separated)" />
      </FormRow>
      <FormRow label="Optimization Epochs:" htmlFor="rl_hop_epochs">
        <NumberInput id="rl_hop_epochs" value={hopEpochs} onChange={e => setHopEpochs(Number(e.target.value))} min={1} max={50} />
      </FormRow>
      <FormRow label="Metric to Optimize:" htmlFor="rl_metric">
        <Select id="rl_metric" value={metric} onChange={e => setMetric(e.target.value)}>
          {Object.entries(HOP_METRICS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>

      <SectionTitle>Ray Tune Framework Settings</SectionTitle>
      <FormRow label="CPU Cores:" htmlFor="rl_cpu_cores">
        <NumberInput id="rl_cpu_cores" value={cpuCores} onChange={e => setCpuCores(Number(e.target.value))} min={1} max={navigator.hardwareConcurrency || 64} />
      </FormRow>
      <FormRow label="Verbose Level (0-3):" htmlFor="rl_verbose">
        <NumberInput id="rl_verbose" value={verbose} onChange={e => setVerbose(Number(e.target.value))} min={0} max={3} />
      </FormRow>
      <FormRow label="Train Best Model:">
        <ToggleButton checked={trainBest} onClick={() => setTrainBest(c => !c)} variant="green-red">
          Train final model with best hyper-parameters
        </ToggleButton>
      </FormRow>
      <FormRow label="Local Mode:">
        <ToggleButton checked={localMode} onClick={() => setLocalMode(c => !c)} variant="red-green">
          Run Ray in Local Mode
        </ToggleButton>
      </FormRow>
      <FormRow label="Number of Samples:" htmlFor="rl_num_samples">
        <NumberInput id="rl_num_samples" value={numSamples} onChange={e => setNumSamples(Number(e.target.value))} min={1} max={1000} />
      </FormRow>

      <SectionTitle>Bayesian Optimization (BO)</SectionTitle>
      <FormRow label="Number of Trials:" htmlFor="rl_n_trials">
        <NumberInput id="rl_n_trials" value={nTrials} onChange={e => setNTrials(Number(e.target.value))} min={1} max={500} />
      </FormRow>
      <CollapsibleSection title="Timeout">
        <FormRow label="Timeout (s):" htmlFor="rl_timeout">
          <Input id="rl_timeout" value={timeout} onChange={e => setTimeout(e.target.value)} placeholder="Timeout in seconds" />
        </FormRow>
      </CollapsibleSection>
      <FormRow label="Startup Trials (before pruning):" htmlFor="rl_n_startup_trials">
        <NumberInput id="rl_n_startup_trials" value={nStartupTrials} onChange={e => setNStartupTrials(Number(e.target.value))} min={0} max={500} />
      </FormRow>
      <FormRow label="Warmup Steps (before pruning):" htmlFor="rl_n_warmup_steps">
        <NumberInput id="rl_n_warmup_steps" value={nWarmupSteps} onChange={e => setNWarmupSteps(Number(e.target.value))} min={0} max={50} />
      </FormRow>
      <FormRow label="Pruning Interval Steps:" htmlFor="rl_interval_steps">
        <NumberInput id="rl_interval_steps" value={intervalSteps} onChange={e => setIntervalSteps(Number(e.target.value))} min={1} max={10} />
      </FormRow>

      <SectionTitle>Distributed Evolutionary Algorithm (DEA)</SectionTitle>
      <FormRow label="Mutation Spread (eta):" htmlFor="rl_eta">
        <NumberInput id="rl_eta" value={eta} onChange={e => setEta(Number(e.target.value))} min={0.01} max={100.0} step={0.5} />
      </FormRow>
      <FormRow label="Gene Mutation Probability (indpb):" htmlFor="rl_indpb">
        <NumberInput id="rl_indpb" value={indpb} onChange={e => setIndpb(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Tournament Size:" htmlFor="rl_tournsize">
        <NumberInput id="rl_tournsize" value={tournsize} onChange={e => setTournsize(Number(e.target.value))} min={2} max={10} />
      </FormRow>
      <FormRow label="Crossover Probability (cxpb):" htmlFor="rl_cxpb">
        <NumberInput id="rl_cxpb" value={cxpb} onChange={e => setCxpb(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Mutation Probability (mutpb):" htmlFor="rl_mutpb">
        <NumberInput id="rl_mutpb" value={mutpb} onChange={e => setMutpb(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Population Size (n_pop):" htmlFor="rl_n_pop">
        <NumberInput id="rl_n_pop" value={nPop} onChange={e => setNPop(Number(e.target.value))} min={1} max={100} />
      </FormRow>
      <FormRow label="Generations (n_gen):" htmlFor="rl_n_gen">
        <NumberInput id="rl_n_gen" value={nGen} onChange={e => setNGen(Number(e.target.value))} min={1} max={100} />
      </FormRow>

      <SectionTitle>Hyperband Optimization (HBO)</SectionTitle>
      <FormRow label="Maximum Trial Resources (timesteps):" htmlFor="rl_max_tres">
        <NumberInput id="rl_max_tres" value={maxTres} onChange={e => setMaxTres(Number(e.target.value))} min={1} max={100} />
      </FormRow>
      <FormRow label="Reduction Factor:" htmlFor="rl_reduction_factor">
        <NumberInput id="rl_reduction_factor" value={reductionFactor} onChange={e => setReductionFactor(Number(e.target.value))} min={2} max={5} />
      </FormRow>
      
      <SectionTitle>Grid Search (GS)</SectionTitle>
      <FormRow label="Grid Search Values:" htmlFor="rl_grid">
        <Input id="rl_grid" value={grid} onChange={e => setGrid(e.target.value)} placeholder="Grid values (space separated floats)" />
      </FormRow>
      <FormRow label="Maximum Concurrent Trials:" htmlFor="rl_max_conc">
        <NumberInput id="rl_max_conc" value={maxConc} onChange={e => setMaxConc(Number(e.target.value))} min={1} max={navigator.hardwareConcurrency || 64} />
      </FormRow>
      
      <SectionTitle>Differential Evolutionary Hyperband (DEHBO)</SectionTitle>
      <FormRow label="Function Evaluations:" htmlFor="rl_fevals">
        <NumberInput id="rl_fevals" value={fevals} onChange={e => setFevals(Number(e.target.value))} min={1} max={1000} />
      </FormRow>
      
      <SectionTitle>Random Search (RS)</SectionTitle>
      <FormRow label="Maximum Trial Failures:" htmlFor="rl_max_failures">
        <NumberInput id="rl_max_failures" value={maxFailures} onChange={e => setMaxFailures(Number(e.target.value))} min={1} max={10} />
      </FormRow>
    </AppTab>
  );
};

export const RLMetaRLTab: React.FC = () => {
  const [mrlMethod, setMrlMethod] = useState('');
  const [mrlHistory, setMrlHistory] = useState(10);
  const [mrlRange, setMrlRange] = useState('0.01 5.0');
  const [mrlExplorationFactor, setMrlExplorationFactor] = useState(2.0);
  const [mrlLr, setMrlLr] = useState(0.001);
  const [tdlLrDecay, setTdlLrDecay] = useState(1.0);
  const [cbExplorationMethod, setCbExplorationMethod] = useState(CB_EXPLORATION_METHODS['Upper Confidence Bound (UCB)']);
  const [cbNumConfigs, setCbNumConfigs] = useState(10);
  const [cbContextFeatures, setCbContextFeatures] = useState('waste overflow length visited_ratio day');
  const [cbFeaturesAggregation, setCbFeaturesAggregation] = useState(AGGREGATION_FUNCTIONS['Average']);
  const [cbEpsilonDecay, setCbEpsilonDecay] = useState(0.995);
  const [cbMinEpsilon, setCbMinEpsilon] = useState(0.01);
  const [morlObjectives, setMorlObjectives] = useState('waste_efficiency overflow_rate');
  const [morlAdaptationRate, setMorlAdaptationRate] = useState(0.1);
  const [rwaModel, setRwaModel] = useState(RWA_MODELS['Recurrent Neural Network (RNN)']);
  const [rwaOptimizer, setRwaOptimizer] = useState(RWA_OPTIMIZERS['Root Mean Square Propagation (RMSProp)']);
  const [rwaEmbeddingDim, setRwaEmbeddingDim] = useState(128);
  const [rwaBatchSize, setRwaBatchSize] = useState(256);
  const [rwaStep, setRwaStep] = useState(100);
  const [rwaUpdateStep, setRwaUpdateStep] = useState(100);

  return (
    <AppTab>
      <SectionTitle className="mt-0">General Settings</SectionTitle>
      <FormRow label="Meta-Learning Method:" htmlFor="rl_mrl_method">
        <Select id="rl_mrl_method" value={mrlMethod} onChange={e => setMrlMethod(e.target.value)}>
          <option value="">Select Method</option>
          {Object.entries(MRL_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="History Length (Days/Epochs):" htmlFor="rl_mrl_history">
        <NumberInput id="rl_mrl_history" value={mrlHistory} onChange={e => setMrlHistory(Number(e.target.value))} min={1} max={100} />
      </FormRow>
      <FormRow label="Dynamic Hyper-Parameter Range:" htmlFor="rl_mrl_range">
        <Input id="rl_mrl_range" value={mrlRange} onChange={e => setMrlRange(e.target.value)} placeholder="Min-Max values (space separated)" />
      </FormRow>
      <FormRow label="Exploration Factor:" htmlFor="rl_mrl_exploration_factor">
        <NumberInput id="rl_mrl_exploration_factor" value={mrlExplorationFactor} onChange={e => setMrlExplorationFactor(Number(e.target.value))} min={0.01} max={10.0} step={0.1} />
      </FormRow>
      <FormRow label="Learning Rate:" htmlFor="rl_mrl_lr">
        <NumberInput id="rl_mrl_lr" value={mrlLr} onChange={e => setMrlLr(Number(e.target.value))} min={1e-6} max={1.0} step={0.0001} />
      </FormRow>
      
      <SectionTitle>Temporal Difference Learning (TDL)</SectionTitle>
      <FormRow label="Learning Rate Decay:" htmlFor="rl_tdl_lr_decay">
        <NumberInput id="rl_tdl_lr_decay" value={tdlLrDecay} onChange={e => setTdlLrDecay(Number(e.target.value))} min={0.0} max={1.0} step={0.001} />
      </FormRow>

      <SectionTitle>Contextual Bandits (CB)</SectionTitle>
      <FormRow label="Exploration Method:" htmlFor="rl_cb_exploration_method">
        <Select id="rl_cb_exploration_method" value={cbExplorationMethod} onChange={e => setCbExplorationMethod(e.target.value)}>
          {Object.entries(CB_EXPLORATION_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Weight Configs:" htmlFor="rl_cb_num_configs">
        <NumberInput id="rl_cb_num_configs" value={cbNumConfigs} onChange={e => setCbNumConfigs(Number(e.target.value))} min={1} max={100} />
      </FormRow>
      <FormRow label="Context Features:" htmlFor="rl_cb_context_features">
        <Input id="rl_cb_context_features" value={cbContextFeatures} onChange={e => setCbContextFeatures(e.target.value)} placeholder="space separated list of features" />
      </FormRow>
      <FormRow label="Feature Aggregation:" htmlFor="rl_cb_features_aggregation">
        <Select id="rl_cb_features_aggregation" value={cbFeaturesAggregation} onChange={e => setCbFeaturesAggregation(e.target.value)}>
          {Object.entries(AGGREGATION_FUNCTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Epsilon Decay:" htmlFor="rl_cb_epsilon_decay">
        <NumberInput id="rl_cb_epsilon_decay" value={cbEpsilonDecay} onChange={e => setCbEpsilonDecay(Number(e.target.value))} min={0.0} max={1.0} step={0.001} />
      </FormRow>
      <FormRow label="Mininimum Epsilon:" htmlFor="rl_cb_min_epsilon">
        <NumberInput id="rl_cb_min_epsilon" value={cbMinEpsilon} onChange={e => setCbMinEpsilon(Number(e.target.value))} min={0.0} max={0.5} step={0.001} />
      </FormRow>
      
      <SectionTitle>Multi-Objective Reinforcement Learning (MORL)</SectionTitle>
      <FormRow label="Objectives:" htmlFor="rl_morl_objectives">
        <Input id="rl_morl_objectives" value={morlObjectives} onChange={e => setMorlObjectives(e.target.value)} placeholder="space separated list of objectives" />
      </FormRow>
      <FormRow label="Adaptation Rate:" htmlFor="rl_morl_adaptation_rate">
        <NumberInput id="rl_morl_adaptation_rate" value={morlAdaptationRate} onChange={e => setMorlAdaptationRate(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      
      <SectionTitle>Reward Weight Adjustment (RWA)</SectionTitle>
      <FormRow label="Model:" htmlFor="rl_rwa_model">
        <Select id="rl_rwa_model" value={rwaModel} onChange={e => setRwaModel(e.target.value)}>
          {Object.entries(RWA_MODELS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Optimizer:" htmlFor="rl_rwa_optimizer">
        <Select id="rl_rwa_optimizer" value={rwaOptimizer} onChange={e => setRwaOptimizer(e.target.value)}>
          {Object.entries(RWA_OPTIMIZERS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Embedding Dim:" htmlFor="rl_rwa_embedding_dim">
        <NumberInput id="rl_rwa_embedding_dim" value={rwaEmbeddingDim} onChange={e => setRwaEmbeddingDim(Number(e.target.value))} min={16} max={512} step={16} />
      </FormRow>
      <FormRow label="Batch Size:" htmlFor="rl_rwa_batch_size">
        <NumberInput id="rl_rwa_batch_size" value={rwaBatchSize} onChange={e => setRwaBatchSize(Number(e.target.value))} min={1} max={1024} step={32} />
      </FormRow>
      <FormRow label="Model Update Step:" htmlFor="rl_rwa_step">
        <NumberInput id="rl_rwa_step" value={rwaStep} onChange={e => setRwaStep(Number(e.target.value))} min={1} max={1000} />
      </FormRow>
      <FormRow label="Weight Update Step:" htmlFor="rl_rwa_update_step">
        <NumberInput id="rl_rwa_update_step" value={rwaUpdateStep} onChange={e => setRwaUpdateStep(Number(e.target.value))} min={1} max={1000} />
      </FormRow>
    </AppTab>
  );
};

export const RLApp: React.FC = () => (
  <AppContainer 
    tabs={[
      { name: 'Training', content: <RLTrainingTab /> },
      { name: 'Model', content: <RLModelTab /> },
      { name: 'Data', content: <RLDataTab /> },
      { name: 'Optimizer', content: <RLOptimizerTab /> },
      { name: 'Costs', content: <RLCostsTab /> },
      { name: 'Output', content: <RLOutputTab /> },
      { name: 'HPO', content: <RLHPOTab /> },
      { name: 'Meta-RL', content: <RLMetaRLTab /> },
    ]}
  />
);