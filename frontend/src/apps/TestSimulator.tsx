// src/apps/test-simulator.tsx
import React, { useState } from 'react';
import { AppTab, AppContainer } from '../components/AppTab';
import {
  FormRow,
  SectionTitle,
  Select,
  NumberInput,
  Input,
  Checkbox,
  ToggleButton,
  CollapsibleSection,
} from '../components/UI';
import {
  SIMULATOR_TEST_POLICIES,
  DATA_DISTRIBUTIONS,
  COUNTY_AREAS,
  WASTE_TYPES,
  DISTANCE_MATRIX_METHODS,
  VERTEX_METHODS,
  EDGE_METHODS,
  DECODE_TYPES,
  PROBLEM_TYPES
} from '../constants';

export const TestSimSettingsTab: React.FC = () => {
  const [selectedPolicies, setSelectedPolicies] = useState<Set<string>>(new Set());
  const [dataDist, setDataDist] = useState(DATA_DISTRIBUTIONS['Gamma 1']);
  const [problem, setProblem] = useState('VRPP');
  const [size, setSize] = useState(50);
  const [days, setDays] = useState(31);
  const [nSamples, setNSamples] = useState(10);
  const [nVehicles, setNVehicles] = useState(1);
  const [seed, setSeed] = useState(42);
  
  const policyKeys = Object.keys(SIMULATOR_TEST_POLICIES);

  const togglePolicy = (policyName: string) => {
    setSelectedPolicies(prev => {
      const next = new Set(prev);
      if (next.has(policyName)) {
        next.delete(policyName);
      } else {
        next.add(policyName);
      }
      return next;
    });
  };

  const selectAll = () => setSelectedPolicies(new Set(policyKeys));
  const deselectAll = () => setSelectedPolicies(new Set());

  return (
    <AppTab>
      <SectionTitle className="mt-0">Test Policies</SectionTitle>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {policyKeys.map(policyName => (
          <ToggleButton
            key={policyName}
            checked={selectedPolicies.has(policyName)}
            onClick={() => togglePolicy(policyName)}
            variant="blue"
          >
            {policyName}
          </ToggleButton>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 mt-2">
        <button
          onClick={selectAll}
          className="w-full px-4 py-2 rounded-md font-medium text-white bg-green-700 hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm"
        >
          Select All
        </button>
        <button
          onClick={deselectAll}
          className="w-full px-4 py-2 rounded-md font-medium text-white bg-red-700 hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm"
        >
          Deselect All
        </button>
      </div>

      <hr className="border-gray-700 my-4" />
      
      <SectionTitle>Test Environment</SectionTitle>
      <FormRow label="Waste Fill Data Distribution:" htmlFor="ts_data_dist">
        <Select id="ts_data_dist" value={dataDist} onChange={e => setDataDist(e.target.value)}>
          {Object.entries(DATA_DISTRIBUTIONS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Problem Type:" htmlFor="ts_problem">
        <Select id="ts_problem" value={problem} onChange={e => setProblem(e.target.value)}>
          {PROBLEM_TYPES.map(p => <option key={p} value={p}>{p}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Graph Size:" htmlFor="ts_size">
        <NumberInput id="ts_size" value={size} onChange={e => setSize(Number(e.target.value))} min={5} max={500} />
      </FormRow>
      <FormRow label="Simulation Days:" htmlFor="ts_days">
        <NumberInput id="ts_days" value={days} onChange={e => setDays(Number(e.target.value))} min={1} max={365} />
      </FormRow>
      <FormRow label="Number of Samples:" htmlFor="ts_n_samples">
        <NumberInput id="ts_n_samples" value={nSamples} onChange={e => setNSamples(Number(e.target.value))} min={1} max={100} />
      </FormRow>
      <FormRow label="Number of Vehicles:" htmlFor="ts_n_vehicles">
        <NumberInput id="ts_n_vehicles" value={nVehicles} onChange={e => setNVehicles(Number(e.target.value))} min={1} max={10} />
      </FormRow>
      <FormRow label="Random Seed:" htmlFor="ts_seed">
        <NumberInput id="ts_seed" value={seed} onChange={e => setSeed(Number(e.target.value))} min={0} max={100000} />
      </FormRow>
    </AppTab>
  );
};

export const TestSimIOTab: React.FC = () => {
  const [outputDir, setOutputDir] = useState('output');
  const [checkpointDir, setCheckpointDir] = useState('temp');
  const [checkpointDays, setCheckpointDays] = useState(5);
  const [wasteFile, setWasteFile] = useState('');
  const [dmFile, setDmFile] = useState('');
  const [binIdxFile, setBinIdxFile] = useState('');
  const [area, setArea] = useState(COUNTY_AREAS['Rio Maior']);
  const [wasteType, setWasteType] = useState('plastic');

  return (
    <AppTab>
      <SectionTitle className="mt-0">Input-Output Paths</SectionTitle>
      <FormRow label="Output Directory:" htmlFor="ts_output_dir">
        <Input id="ts_output_dir" value={outputDir} onChange={e => setOutputDir(e.target.value)} />
      </FormRow>
      <FormRow label="Checkpoint Directory:" htmlFor="ts_checkpoint_dir">
        <Input id="ts_checkpoint_dir" value={checkpointDir} onChange={e => setCheckpointDir(e.target.value)} />
      </FormRow>
      <FormRow label="Checkpoint Save Days:" htmlFor="ts_checkpoint_days">
        <NumberInput id="ts_checkpoint_days" value={checkpointDays} onChange={e => setCheckpointDays(Number(e.target.value))} min={0} max={365} />
      </FormRow>

      <SectionTitle>Input Files</SectionTitle>
      <FormRow label="Waste Fill File:" htmlFor="ts_waste_file">
        <Input id="ts_waste_file" value={wasteFile} onChange={e => setWasteFile(e.target.value)} placeholder="path/to/waste_file.pkl" />
      </FormRow>
      <FormRow label="Distance Matrix File:" htmlFor="ts_dm_file">
        <Input id="ts_dm_file" value={dmFile} onChange={e => setDmFile(e.target.value)} placeholder="path/to/dm_file.pkl" />
      </FormRow>
      <FormRow label="Bin Index File:" htmlFor="ts_bin_idx_file">
        <Input id="ts_bin_idx_file" value={binIdxFile} onChange={e => setBinIdxFile(e.target.value)} placeholder="path/to/bin_idx_file.pkl" />
      </FormRow>

      <SectionTitle>Simulator Data Context</SectionTitle>
      <FormRow label="County Area:" htmlFor="ts_area">
        <Select id="ts_area" value={area} onChange={e => setArea(e.target.value)}>
          {Object.entries(COUNTY_AREAS).filter(([k]) => k === 'Rio Maior' || k === 'Lisbon' || k === 'Porto').map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Waste Type:" htmlFor="ts_waste_type">
        <Select id="ts_waste_type" value={wasteType} onChange={e => setWasteType(e.target.value)}>
          {WASTE_TYPES.map(w => <option key={w} value={w.toLowerCase()}>{w}</option>)}
        </Select>
      </FormRow>
    </AppTab>
  );
};

export const TestSimPolicyParamsTab: React.FC = () => {
  const [decodeType, setDecodeType] = useState('greedy');
  const [temperature, setTemperature] = useState(1.0);
  const [pregularLevel, setPregularLevel] = useState('');
  const [plastminuteCf, setPlastminuteCf] = useState('');
  const [gurobiParam, setGurobiParam] = useState('');
  const [hexalyParam, setHexalyParam] = useState('');
  const [lookaheadA, setLookaheadA] = useState(false);
  const [lookaheadB, setLookaheadB] = useState(false);
  const [cacheRegular, setCacheRegular] = useState(false);
  const [runTsp, setRunTsp] = useState(false);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Model Parameters</SectionTitle>
      <FormRow label="Decode Type:" htmlFor="ts_decode_type">
        <Select id="ts_decode_type" value={decodeType} onChange={e => setDecodeType(e.target.value)}>
          {DECODE_TYPES.map(dt => <option key={dt} value={dt}>{dt.charAt(0).toUpperCase() + dt.slice(1)}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Softmax Temperature:" htmlFor="ts_temperature">
        <NumberInput id="ts_temperature" value={temperature} onChange={e => setTemperature(Number(e.target.value))} min={0.0} max={5.0} step={0.1} />
      </FormRow>

      <SectionTitle>Policy Parameters (Space-separated lists)</SectionTitle>
      <FormRow label="Regular Policy Level:" htmlFor="ts_pregular">
        <Input id="ts_pregular" value={pregularLevel} onChange={e => setPregularLevel(e.target.value)} placeholder="e.g., 2 3 6 (for --lvl)" />
      </FormRow>
      <FormRow label="Last Minute CF:" htmlFor="ts_plastminute">
        <Input id="ts_plastminute" value={plastminuteCf} onChange={e => setPlastminuteCf(e.target.value)} placeholder="e.g., 50 70 90 (for --cf)" />
      </FormRow>
      <FormRow label="Gurobi VRPP Parameter:" htmlFor="ts_gurobi">
        <Input id="ts_gurobi" value={gurobiParam} onChange={e => setGurobiParam(e.target.value)} placeholder="e.g., 0.42 0.84 (for --gp)" />
      </FormRow>
      <FormRow label="Hexaly VRPP Parameter:" htmlFor="ts_hexaly">
        <Input id="ts_hexaly" value={hexalyParam} onChange={e => setHexalyParam(e.target.value)} placeholder="e.g., 0.42 0.84 (for --hp)" />
      </FormRow>

      <SectionTitle>Boolean Flags</SectionTitle>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Checkbox id="ts_lookahead_a" label="Look-Ahead Configuration A" checked={lookaheadA} onChange={e => setLookaheadA(e.target.checked)} />
        <Checkbox id="ts_lookahead_b" label="Look-Ahead Configuration B" checked={lookaheadB} onChange={e => setLookaheadB(e.target.checked)} />
        <Checkbox id="ts_cache_regular" label="Deactivate Regular Cache" checked={cacheRegular} onChange={e => setCacheRegular(e.target.checked)} />
        <Checkbox id="ts_run_tsp" label="Run fast_tsp for routing" checked={runTsp} onChange={e => setRunTsp(e.target.checked)} />
      </div>
    </AppTab>
  );
};

export const TestSimAdvancedTab: React.FC = () => {
  const [cpuCores, setCpuCores] = useState(navigator.hardwareConcurrency || 4);
  const [envFile, setEnvFile] = useState('vars.env');
  const [serverRun, setServerRun] = useState(false);
  const [progressBar, setProgressBar] = useState(true); // 'no_progress_check' is False
  const [resume, setResume] = useState(false);
  const [vertexMethod, setVertexMethod] = useState(VERTEX_METHODS['Min-Max Normalization']);
  const [distanceMethod, setDistanceMethod] = useState(DISTANCE_MATRIX_METHODS['Google Maps (GMaps)']);
  const [edgeThreshold, setEdgeThreshold] = useState(1.0);
  const [edgeMethod, setEdgeMethod] = useState(EDGE_METHODS['K-Nearest Neighbors (KNN)']);
  const [gplicFile, setGplicFile] = useState('');
  const [hexlicFile, setHexlicFile] = useState('');
  const [gapikFile, setGapikFile] = useState('');
  const [symkeyName, setSymkeyName] = useState('');

  return (
    <AppTab>
      <SectionTitle className="mt-0">System Settings</SectionTitle>
      <FormRow label="Maximum CPU Cores:" htmlFor="ts_cpu_cores">
        <NumberInput id="ts_cpu_cores" value={cpuCores} onChange={e => setCpuCores(Number(e.target.value))} min={1} max={navigator.hardwareConcurrency || 64} />
      </FormRow>
      <FormRow label="Environment Variables File:" htmlFor="ts_env_file">
        <Input id="ts_env_file" value={envFile} onChange={e => setEnvFile(e.target.value)} />
      </FormRow>
      <FormRow label="Flags:">
        <div className="space-y-3">
          <ToggleButton checked={serverRun} onClick={() => setServerRun(c => !c)} variant="red-green">
            Remote Server Execution
          </ToggleButton>
          <ToggleButton checked={progressBar} onClick={() => setProgressBar(c => !c)} variant="green-red">
            Progress Bar
          </ToggleButton>
          <ToggleButton checked={resume} onClick={() => setResume(c => !c)} variant="red-green">
            Resume Testing
          </ToggleButton>
        </div>
      </FormRow>

      <SectionTitle>Edge/Vertex Setup</SectionTitle>
      <FormRow label="Vertex Method:" htmlFor="ts_vertex_method">
        <Select id="ts_vertex_method" value={vertexMethod} onChange={e => setVertexMethod(e.target.value)}>
          {Object.entries(VERTEX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Distance Method:" htmlFor="ts_distance_method">
        <Select id="ts_distance_method" value={distanceMethod} onChange={e => setDistanceMethod(e.target.value)}>
          {Object.entries(DISTANCE_MATRIX_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>
      <FormRow label="Edge Threshold (0.0 to 1.0):" htmlFor="ts_edge_threshold">
        <NumberInput id="ts_edge_threshold" value={edgeThreshold} onChange={e => setEdgeThreshold(Number(e.target.value))} min={0.0} max={1.0} step={0.01} />
      </FormRow>
      <FormRow label="Edge Method:" htmlFor="ts_edge_method">
        <Select id="ts_edge_method" value={edgeMethod} onChange={e => setEdgeMethod(e.target.value)}>
          {Object.entries(EDGE_METHODS).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
        </Select>
      </FormRow>

      <CollapsibleSection title="Key/License Files" className="mt-4">
        <FormRow label="Gurobi License File:" htmlFor="ts_gplic">
          <Input id="ts_gplic" value={gplicFile} onChange={e => setGplicFile(e.target.value)} />
        </FormRow>
        <FormRow label="Hexaly License File:" htmlFor="ts_hexlic">
          <Input id="ts_hexlic" value={hexlicFile} onChange={e => setHexlicFile(e.target.value)} />
        </FormRow>
        <FormRow label="Google API Key File:" htmlFor="ts_gapik">
          <Input id="ts_gapik" value={gapikFile} onChange={e => setGapikFile(e.target.value)} />
        </FormRow>
        <FormRow label="Cryptographic Key Name:" htmlFor="ts_symkey">
          <Input id="ts_symkey" value={symkeyName} onChange={e => setSymkeyName(e.target.value)} />
        </FormRow>
      </CollapsibleSection>
    </AppTab>
  );
};

export const TestSimulatorApp: React.FC = () => (
  <AppContainer 
    tabs={[
      { name: 'Settings', content: <TestSimSettingsTab /> },
      { name: 'I/O', content: <TestSimIOTab /> },
      { name: 'Policy Params', content: <TestSimPolicyParamsTab /> },
      { name: 'Advanced', content: <TestSimAdvancedTab /> },
    ]} 
  />
);