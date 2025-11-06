// src/apps/generate-data.tsx
import React, { useState } from 'react';
import { AppTab, AppContainer } from '../components/AppTab';
import {
  FormRow,
  SectionTitle,
  Input,
  NumberInput,
  Select,
  Checkbox,
  ToggleButton,
  CollapsibleSection,
} from '../components/UI';
import {
  PROBLEM_TYPES,
  DATA_DISTRIBUTIONS,
  VERTEX_METHODS,
} from '../constants';

export const GenDataGeneralTab: React.FC = () => {
  const [name, setName] = useState('default_dataset');
  const [filename, setFilename] = useState('');
  const [dataDir, setDataDir] = useState('datasets');
  const [datasetSize, setDatasetSize] = useState(128000);
  const [datasetType, setDatasetType] = useState('train');
  const [seed, setSeed] = useState(42);
  const [overwrite, setOverwrite] = useState(false);

  return (
    <AppTab>
      <FormRow label="Dataset Name:" htmlFor="gd_name">
        <Input id="gd_name" value={name} onChange={e => setName(e.target.value)} />
      </FormRow>
      <CollapsibleSection title="Filename">
        <FormRow label="Specific Filename:" htmlFor="gd_filename">
          <Input id="gd_filename" value={filename} onChange={e => setFilename(e.target.value)} placeholder="e.g., my_data.pkl (ignores data_dir)" />
        </FormRow>
      </CollapsibleSection>
      <FormRow label="Data Directory:" htmlFor="gd_data_dir">
        <Input id="gd_data_dir" value={dataDir} onChange={e => setDataDir(e.target.value)} />
      </FormRow>
      <FormRow label="Dataset Size:" htmlFor="gd_dataset_size">
        <NumberInput id="gd_dataset_size" value={datasetSize} onChange={e => setDatasetSize(Number(e.target.value))} min={1000} max={10000000} step={1000} />
      </FormRow>
      <FormRow label="Dataset Type:" htmlFor="gd_dataset_type">
        <Select id="gd_dataset_type" value={datasetType} onChange={e => setDatasetType(e.target.value)}>
          <option value="train">train</option>
          <option value="train_time">train_time</option>
          <option value="test_simulator">test_simulator</option>
        </Select>
      </FormRow>
      <FormRow label="Random Seed:" htmlFor="gd_seed">
        <NumberInput id="gd_seed" value={seed} onChange={e => setSeed(Number(e.target.value))} min={0} max={100000} />
      </FormRow>
      <FormRow label="Options:">
        <Checkbox id="gd_overwrite" label="Overwrite existing file" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
      </FormRow>
    </AppTab>
  );
};

export const GenDataProblemTab: React.FC = () => {
  const [problem, setProblem] = useState('All');
  const [graphSizes, setGraphSizes] = useState('20 50 100');
  const [selectedDists, setSelectedDists] = useState<Set<string>>(new Set(['All']));
  const [isGaussian, setIsGaussian] = useState(false);
  const [sigma, setSigma] = useState(0.6);
  const [penaltyFactor, setPenaltyFactor] = useState(3.0);

  const distKeys = Object.keys(DATA_DISTRIBUTIONS);

  const toggleDist = (distName: string) => {
    setSelectedDists(prev => {
      const next = new Set(prev);
      if (next.has(distName)) {
        next.delete(distName);
      } else {
        next.add(distName);
      }
      return next;
    });
  };

  const selectAllDists = () => setSelectedDists(new Set(distKeys));
  const deselectAllDists = () => setSelectedDists(new Set());

  return (
    <AppTab>
      <FormRow label="Problem Type:" htmlFor="gd_problem">
        <Select id="gd_problem" value={problem} onChange={e => setProblem(e.target.value)}>
          {[...PROBLEM_TYPES, 'All'].map(p => <option key={p} value={p}>{p}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Graph Sizes:" htmlFor="gd_graph_sizes">
        <Input id="gd_graph_sizes" value={graphSizes} onChange={e => setGraphSizes(e.target.value)} placeholder="Space-separated list of sizes (e.g., 20 50 100)" />
      </FormRow>

      <SectionTitle>Data Distributions:</SectionTitle>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {distKeys.map(distName => (
          <ToggleButton
            key={distName}
            checked={selectedDists.has(distName)}
            onClick={() => toggleDist(distName)}
            variant="blue"
          >
            {distName}
          </ToggleButton>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 mt-2">
        <button onClick={selectAllDists} className="w-full px-4 py-2 rounded-md font-medium text-white bg-green-700 hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm">
          Select All
        </button>
        <button onClick={deselectAllDists} className="w-full px-4 py-2 rounded-md font-medium text-white bg-red-700 hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm">
          Deselect All
        </button>
      </div>

      <SectionTitle>PDP Parameters</SectionTitle>
      <FormRow label="Gaussian:">
        <ToggleButton checked={isGaussian} onClick={() => setIsGaussian(c => !c)} variant="red-green">
          Use Gaussian Distribution
        </ToggleButton>
      </FormRow>
      <FormRow label="Sigma Value:" htmlFor="gd_sigma">
        <NumberInput id="gd_sigma" value={sigma} onChange={e => setSigma(Number(e.target.value))} min={0} max={1} step={0.1} disabled={!isGaussian} />
      </FormRow>

      <SectionTitle>PCTSP Parameters</SectionTitle>
      <FormRow label="Penalty Factor:" htmlFor="gd_penalty">
        <NumberInput id="gd_penalty" value={penaltyFactor} onChange={e => setPenaltyFactor(Number(e.target.value))} min={0.1} max={10.0} step={0.1} />
      </FormRow>
    </AppTab>
  );
};

export const GenDataAdvancedTab: React.FC = () => {
  const [nEpochs, setNEpochs] = useState(1);
  const [epochStart, setEpochStart] = useState(0);
  const [vertexMethod, setVertexMethod] = useState(VERTEX_METHODS['Min-Max Normalization']);
  const [area, setArea] = useState('');
  const [wasteType, setWasteType] = useState('');
  const [focusGraphs, setFocusGraphs] = useState('');
  const [focusSize, setFocusSize] = useState(0);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Epoch Settings</SectionTitle>
      <FormRow label="Number of Epochs:" htmlFor="gd_n_epochs">
        <NumberInput id="gd_n_epochs" value={nEpochs} onChange={e => setNEpochs(Number(e.target.value))} min={1} max={1000} />
      </FormRow>
      <FormRow label="Start Epoch:" htmlFor="gd_epoch_start">
        <NumberInput id="gd_epoch_start" value={epochStart} onChange={e => setEpochStart(Number(e.target.value))} min={0} max={1000} />
      </FormRow>
      <FormRow label="Vertex Method:" htmlFor="gd_vertex_method">
        <Select id="gd_vertex_method" value={vertexMethod} onChange={e => setVertexMethod(e.target.value)}>
          {Object.entries(VERTEX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>

      <CollapsibleSection title="Area Specific" className="mt-4">
        <FormRow label="County Area:" htmlFor="gd_area">
          <Input id="gd_area" value={area} onChange={e => setArea(e.target.value)} placeholder="Rio Maior" />
        </FormRow>
        <FormRow label="Waste Type:" htmlFor="gd_waste_type">
          <Input id="gd_waste_type" value={wasteType} onChange={e => setWasteType(e.target.value)} placeholder="Plastic" />
        </FormRow>
      </CollapsibleSection>

      <CollapsibleSection title="Focus Graphs" className="mt-4">
        <FormRow label="Focus Graph Paths:" htmlFor="gd_focus_graphs">
          <Input id="gd_focus_graphs" value={focusGraphs} onChange={e => setFocusGraphs(e.target.value)} placeholder="Paths to focus graph files" />
        </FormRow>
        <FormRow label="Number per Focus Graph:" htmlFor="gd_focus_size">
          <NumberInput id="gd_focus_size" value={focusSize} onChange={e => setFocusSize(Number(e.target.value))} min={0} max={100000} />
        </FormRow>
      </CollapsibleSection>
    </AppTab>
  );
};

export const GenerateDataApp: React.FC = () => (
  <AppContainer
    tabs={[
      { name: 'General', content: <GenDataGeneralTab /> },
      { name: 'Problem', content: <GenDataProblemTab /> },
      { name: 'Advanced', content: <GenDataAdvancedTab /> },
    ]}
  />
);