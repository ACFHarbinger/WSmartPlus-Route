// src/apps/evaluation.tsx
import React, { useState } from 'react';
import { AppTab, AppContainer } from '../components/AppTab';
import {
  FormRow,
  SectionTitle,
  Input,
  NumberInput,
  Select,
  ToggleButton,
  CollapsibleSection,
  Checkbox,
} from '../components/UI';
import {
  DECODE_STRATEGIES,
  DECODE_TYPES,
  DISTANCE_MATRIX_METHODS,
  VERTEX_METHODS,
  EDGE_METHODS,
} from '../constants';

export const EvalIOTab: React.FC = () => {
  const [datasets, setDatasets] = useState('');
  const [model, setModel] = useState('');
  const [resultsDir, setResultsDir] = useState('results');
  const [outputFile, setOutputFile] = useState('');
  const [overwrite, setOverwrite] = useState(false);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Input Files</SectionTitle>
      <FormRow label="Dataset Files:" htmlFor="eval_datasets">
        <Input id="eval_datasets" value={datasets} onChange={e => setDatasets(e.target.value)} placeholder="e.g., dataset_a.pkl dataset_b.pkl (space separated)" />
      </FormRow>
      <FormRow label="Model Checkpoint:" htmlFor="eval_model">
        <Input id="eval_model" value={model} onChange={e => setModel(e.target.value)} placeholder="Path to trained model checkpoint" />
      </FormRow>

      <SectionTitle>Output Settings</SectionTitle>
      <FormRow label="Results Directory:" htmlFor="eval_results_dir">
        <Input id="eval_results_dir" value={resultsDir} onChange={e => setResultsDir(e.target.value)} />
      </FormRow>

      <CollapsibleSection title="Output File">
        <FormRow label="Output File Name:" htmlFor="eval_output_file">
          <Input id="eval_output_file" value={outputFile} onChange={e => setOutputFile(e.target.value)} placeholder="Results file name" />
        </FormRow>
        <FormRow label="Options:">
          <Checkbox id="eval_overwrite" label="Overwrite existing results file" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
        </FormRow>
      </CollapsibleSection>
    </AppTab>
  );
};

export const EvalProblemTab: React.FC = () => {
  const [graphSize, setGraphSize] = useState(50);
  const [area, setArea] = useState('riomaior');
  const [wasteType, setWasteType] = useState('plastic');
  const [focusGraph, setFocusGraph] = useState('');
  const [focusSize, setFocusSize] = useState(0);
  const [distanceMethod, setDistanceMethod] = useState(DISTANCE_MATRIX_METHODS['Google Maps (GMaps)']);
  const [vertexMethod, setVertexMethod] = useState(VERTEX_METHODS['Min-Max Normalization']);
  const [edgeThreshold, setEdgeThreshold] = useState(1.0);
  const [edgeMethod, setEdgeMethod] = useState(EDGE_METHODS['K-Nearest Neighbors (KNN)']);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Problem Instance</SectionTitle>
      <FormRow label="Graph Size:" htmlFor="eval_graph_size">
        <NumberInput id="eval_graph_size" value={graphSize} onChange={e => setGraphSize(Number(e.target.value))} min={1} max={500} />
      </FormRow>
      <FormRow label="County Area:" htmlFor="eval_area">
        <Input id="eval_area" value={area} onChange={e => setArea(e.target.value)} />
      </FormRow>
      <FormRow label="Waste Type:" htmlFor="eval_waste_type">
        <Input id="eval_waste_type" value={wasteType} onChange={e => setWasteType(e.target.value)} />
      </FormRow>

      <CollapsibleSection title="Focus Graph">
        <FormRow label="Focus Graph Path:" htmlFor="eval_focus_graph">
          <Input id="eval_focus_graph" value={focusGraph} onChange={e => setFocusGraph(e.target.value)} placeholder="Path to focus graph file" />
        </FormRow>
        <FormRow label="Number of Focus Graphs:" htmlFor="eval_focus_size">
          <NumberInput id="eval_focus_size" value={focusSize} onChange={e => setFocusSize(Number(e.target.value))} min={0} max={1000} />
        </FormRow>
      </CollapsibleSection>

      <SectionTitle>Preprocessing Methods</SectionTitle>
      <FormRow label="Distance Method:" htmlFor="eval_distance_method">
        <Select id="eval_distance_method" value={distanceMethod} onChange={e => setDistanceMethod(e.target.value)}>
          {Object.entries(DISTANCE_MATRIX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Vertex Method:" htmlFor="eval_vertex_method">
        <Select id="eval_vertex_method" value={vertexMethod} onChange={e => setVertexMethod(e.target.value)}>
          {Object.entries(VERTEX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Edge Threshold (0.0 to 1.0):" htmlFor="eval_edge_threshold">
        <NumberInput id="eval_edge_threshold" value={edgeThreshold} onChange={e => setEdgeThreshold(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Edge Method:" htmlFor="eval_edge_method">
        <Select id="eval_edge_method" value={edgeMethod} onChange={e => setEdgeMethod(e.target.value)}>
          {Object.entries(EDGE_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
    </AppTab>
  );
};

export const EvalDecodingTab: React.FC = () => {
  const [decodeStrategy, setDecodeStrategy] = useState(DECODE_STRATEGIES['Greedy']);
  const [decodeType, setDecodeType] = useState('greedy');
  const [width, setWidth] = useState('');
  const [softmaxTemp, setSoftmaxTemp] = useState(1.0);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Decoding Parameters</SectionTitle>
      <FormRow label="Decode Strategy:" htmlFor="eval_decode_strategy">
        <Select id="eval_decode_strategy" value={decodeStrategy} onChange={e => setDecodeStrategy(e.target.value)}>
          {Object.entries(DECODE_STRATEGIES).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Decode Type (Output):" htmlFor="eval_decode_type">
        <Select id="eval_decode_type" value={decodeType} onChange={e => setDecodeType(e.target.value)}>
          {DECODE_TYPES.map(dt => <option key={dt} value={dt}>{dt.charAt(0).toUpperCase() + dt.slice(1)}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Beam Width / Samples:" htmlFor="eval_width">
        <Input id="eval_width" value={width} onChange={e => setWidth(e.target.value)} placeholder="e.g., 50 100 200 (space separated)" />
      </FormRow>
      <FormRow label="Softmax Temperature:" htmlFor="eval_softmax_temp">
        <NumberInput id="eval_softmax_temp" value={softmaxTemp} onChange={e => setSoftmaxTemp(Number(e.target.value))} min={0.01} max={10.0} step={0.1} />
      </FormRow>
    </AppTab>
  );
};

export const EvalDataBatchingTab: React.FC = () => {
  const [valSize, setValSize] = useState(12800);
  const [offset, setOffset] = useState(0);
  const [evalBatchSize, setEvalBatchSize] = useState(256);
  const [maxCalcBatchSize, setMaxCalcBatchSize] = useState(12800);
  const [useCuda, setUseCuda] = useState(true); // 'no_cuda' is False
  const [useProgressBar, setUseProgressBar] = useState(true); // 'no_progress_bar' is False
  const [compressMask, setCompressMask] = useState(false);
  const [multiprocessing, setMultiprocessing] = useState(false);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Data Processing</SectionTitle>
      <FormRow label="Validation Size:" htmlFor="eval_val_size">
        <NumberInput id="eval_val_size" value={valSize} onChange={e => setValSize(Number(e.target.value))} min={1} max={100000} step={100} />
      </FormRow>
      <FormRow label="Dataset Offset:" htmlFor="eval_offset">
        <NumberInput id="eval_offset" value={offset} onChange={e => setOffset(Number(e.target.value))} min={0} max={100000} />
      </FormRow>
      <FormRow label="Evaluation Batch Size:" htmlFor="eval_eval_batch_size">
        <NumberInput id="eval_eval_batch_size" value={evalBatchSize} onChange={e => setEvalBatchSize(Number(e.target.value))} min={1} max={1024} />
      </FormRow>
      <FormRow label="Maximum Calculation Sub-Batch Size:" htmlFor="eval_max_calc_batch_size">
        <NumberInput id="eval_max_calc_batch_size" value={maxCalcBatchSize} onChange={e => setMaxCalcBatchSize(Number(e.target.value))} min={1} max={100000} step={100} />
      </FormRow>
      
      <SectionTitle>System Flags</SectionTitle>
      <FormRow label="CUDA:">
        <ToggleButton checked={useCuda} onClick={() => setUseCuda(c => !c)} variant="green-red">
          Use CUDA (Nvidia GPU)
        </ToggleButton>
      </FormRow>
      <FormRow label="Progress Bar:">
        <ToggleButton checked={useProgressBar} onClick={() => setUseProgressBar(c => !c)} variant="green-red">
          Show Progress Bar
        </ToggleButton>
      </FormRow>
      <FormRow label="Compress Mask:">
        <ToggleButton checked={compressMask} onClick={() => setCompressMask(c => !c)} variant="red-green">
          Compress Mask
        </ToggleButton>
      </FormRow>
      <FormRow label="Multiprocessing:">
        <ToggleButton checked={multiprocessing} onClick={() => setMultiprocessing(c => !c)} variant="red-green">
          Use Multiprocessing
        </ToggleButton>
      </FormRow>
    </AppTab>
  );
};

export const EvaluationApp: React.FC = () => (
  <AppContainer 
    tabs={[
      { name: 'I/O', content: <EvalIOTab /> },
      { name: 'Problem', content: <EvalProblemTab /> },
      { name: 'Decoding', content: <EvalDecodingTab /> },
      { name: 'Data/Batching', content: <EvalDataBatchingTab /> },
    ]}
  />
);