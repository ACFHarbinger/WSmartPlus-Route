// src/apps/file-system-tools.tsx
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
import { SCRIPTS, OPERATION_MAP, FUNCTION_MAP } from '../constants';

export const FileSystemScriptsTab: React.FC = () => {
  const [selectedScript, setSelectedScript] = useState<string | null>(null);
  const [verbose, setVerbose] = useState(false);
  const [cores, setCores] = useState(navigator.hardwareConcurrency || 4);
  const [envManager, setEnvManager] = useState('uv');
  
  // Note: Platform-specific logic (like for Slurm) is simplified to always show.
  const allScriptKeys = Object.keys(SCRIPTS);
  const linuxOnlyScripts = ["slurm", "slim_slurm"];
  
  const selectScript = (scriptName: string) => {
    setSelectedScript(prev => prev === scriptName ? null : scriptName);
  };
  
  const clearSelection = () => setSelectedScript(null);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Script Selection</SectionTitle>
      <div className="grid grid-cols-2 gap-2">
        {allScriptKeys.filter(k => !linuxOnlyScripts.includes(k)).map(scriptName => (
          <ToggleButton
            key={scriptName}
            checked={selectedScript === scriptName}
            onClick={() => selectScript(scriptName)}
            variant="blue"
            className="min-h-[38px]"
          >
            {SCRIPTS[scriptName as keyof typeof SCRIPTS]}
          </ToggleButton>
        ))}
      </div>
      <button onClick={clearSelection} className="w-full mt-2 px-4 py-2 rounded-md font-medium text-white bg-red-700 hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm">
        Clear Selection
      </button>

      <CollapsibleSection title="Linux Only Scripts" className="mt-4">
        <div className="grid grid-cols-2 gap-2">
          {linuxOnlyScripts.map(scriptName => (
            <ToggleButton
              key={scriptName}
              checked={selectedScript === scriptName}
              onClick={() => selectScript(scriptName)}
              variant="blue"
              className="min-h-[38px] bg-yellow-600 text-red-900 font-bold hover:bg-yellow-500"
            >
              {SCRIPTS[scriptName as keyof typeof SCRIPTS]} (Linux Only)
            </ToggleButton>
          ))}
        </div>
      </CollapsibleSection>

      <hr className="border-gray-700 my-4" />
      <SectionTitle>Script Parameters</SectionTitle>
      <FormRow label="Selected Script:">
        <div className="px-3 py-2 bg-gray-700 rounded-md text-gray-200 font-medium">
          {selectedScript ? SCRIPTS[selectedScript as keyof typeof SCRIPTS] : "No script selected"}
        </div>
      </FormRow>
      <FormRow label="Verbose Output:">
        <ToggleButton checked={verbose} onClick={() => setVerbose(c => !c)} variant="red-green">
          Verbose Mode
        </ToggleButton>
      </FormRow>
      <FormRow label="CPU Cores:" htmlFor="fs_cores">
        <NumberInput id="fs_cores" value={cores} onChange={e => setCores(Number(e.target.value))} min={1} max={navigator.hardwareConcurrency || 64} />
      </FormRow>

      {selectedScript === 'setup_env' && (
        <>
          <SectionTitle>Script-Specific Parameters</SectionTitle>
          <FormRow label="Package Manager:" htmlFor="fs_env_manager">
            <Select id="fs_env_manager" value={envManager} onChange={e => setEnvManager(e.target.value)}>
              <option value="uv">uv</option>
              <option value="conda">conda</option>
              <option value="venv">venv</option>
            </Select>
          </FormRow>
        </>
      )}
      {selectedScript && selectedScript !== 'setup_env' && (
         <>
          <SectionTitle>Script-Specific Parameters</SectionTitle>
          <p className="text-gray-400 text-sm md:col-span-3">No specific parameters for this script.</p>
        </>
      )}
    </AppTab>
  );
};

export const FileSystemUpdateTab: React.FC = () => {
  const [targetEntry, setTargetEntry] = useState('');
  const [outputKey, setOutputKey] = useState('');
  const [filenamePattern, setFilenamePattern] = useState('');
  const [preview, setPreview] = useState(true);
  const [updateOp, setUpdateOp] = useState('');
  const [updateValue, setUpdateValue] = useState('0.0');
  const [inputKey1, setInputKey1] = useState('');
  const [inputKey2, setInputKey2] = useState('');
  const [statsFunc, setStatsFunc] = useState('');
  const [outputFilename, setOutputFilename] = useState('');

  return (
    <AppTab>
      <SectionTitle className="mt-0">File Targeting</SectionTitle>
      <FormRow label="Target Entry Path:" htmlFor="fs_target_entry">
        <Input id="fs_target_entry" value={targetEntry} onChange={e => setTargetEntry(e.target.value)} placeholder="e.g., /path/to/directory or /path/to/file.json" />
      </FormRow>
      <FormRow label="Output Field Key:" htmlFor="fs_output_key">
        <Input id="fs_output_key" value={outputKey} onChange={e => setOutputKey(e.target.value)} placeholder="e.g., 'overflows' (or the calculated output key)" />
      </FormRow>
      <FormRow label="Glob Filename Pattern:" htmlFor="fs_filename_pattern">
        <Input id="fs_filename_pattern" value={filenamePattern} onChange={e => setFilenamePattern(e.target.value)} placeholder="e.g., 'log_*.json'" />
      </FormRow>
      <hr className="border-gray-700 my-4" />
      <FormRow label="Preview Mode:">
        <ToggleButton checked={preview} onClick={() => setPreview(c => !c)} variant="red-green">
          {preview ? "Preview Update (Safe)" : "Perform Update (Dangerous)"}
        </ToggleButton>
      </FormRow>

      <CollapsibleSection title="Inplace Update Parameters" className="mt-4">
        <FormRow label="Update Operation:" htmlFor="fs_update_op">
          <Select id="fs_update_op" value={updateOp} onChange={e => setUpdateOp(e.target.value)}>
            <option value="">Select Operation</option>
            {Object.entries(OPERATION_MAP).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
          </Select>
        </FormRow>
        <FormRow label="Update Value (single-input):" htmlFor="fs_update_value">
          <Input id="fs_update_value" value={updateValue} onChange={e => setUpdateValue(e.target.value)} placeholder="Enter a float or string value" />
        </FormRow>
        <FormRow label="Input Keys (two-input):">
          <div className="grid grid-cols-2 gap-2">
            <Input value={inputKey1} onChange={e => setInputKey1(e.target.value)} placeholder="Key 1 (e.g., 'total_miles')" />
            <Input value={inputKey2} onChange={e => setInputKey2(e.target.value)} placeholder="Key 2 (e.g., 'cost_per_mile')" />
          </div>
        </FormRow>
      </CollapsibleSection>

      <CollapsibleSection title="Statistics Update Parameters" className="mt-4">
        <FormRow label="Update Function:" htmlFor="fs_stats_func">
          <Select id="fs_stats_func" value={statsFunc} onChange={e => setStatsFunc(e.target.value)}>
            <option value="">Select Statistics Function</option>
            {Object.entries(FUNCTION_MAP).map(([name, value]) => <option key={value} value={value}>{name}</option>)}
          </Select>
        </FormRow>
        <FormRow label="Output Filename:" htmlFor="fs_output_filename">
          <Input id="fs_output_filename" value={outputFilename} onChange={e => setOutputFilename(e.target.value)} placeholder="Name of output file (e.g., log_mean.json)" />
        </FormRow>
      </CollapsibleSection>
    </AppTab>
  );
};

export const FileSystemDeleteTab: React.FC = () => {
  const [logDir, setLogDir] = useState('logs');
  const [outputDir, setOutputDir] = useState('model_weights');
  const [dataDir, setDataDir] = useState('datasets');
  const [evalDir, setEvalDir] = useState('results');
  const [testDir, setTestDir] = useState('output');
  const [testCheckpointDir, setTestCheckpointDir] = useState('temp');
  const [keepWandb, setKeepWandb] = useState(true);
  const [keepLog, setKeepLog] = useState(true);
  const [keepOutput, setKeepOutput] = useState(true);
  const [deleteData, setDeleteData] = useState(false);
  const [deleteEval, setDeleteEval] = useState(false);
  const [deleteTestSim, setDeleteTestSim] = useState(false);
  const [deleteTestSimCheckpoint, setDeleteTestSimCheckpoint] = useState(false);
  const [deleteCache, setDeleteCache] = useState(false);
  const [preview, setPreview] = useState(true);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Directory Paths</SectionTitle>
      <FormRow label="Train Log Directory:" htmlFor="fs_log_dir">
        <Input id="fs_log_dir" value={logDir} onChange={e => setLogDir(e.target.value)} />
      </FormRow>
      <FormRow label="Output Models Directory:" htmlFor="fs_output_dir">
        <Input id="fs_output_dir" value={outputDir} onChange={e => setOutputDir(e.target.value)} />
      </FormRow>
      <FormRow label="Datasets Directory:" htmlFor="fs_data_dir">
        <Input id="fs_data_dir" value={dataDir} onChange={e => setDataDir(e.target.value)} />
      </FormRow>
      <FormRow label="Evaluation Results Directory:" htmlFor="fs_eval_dir">
        <Input id="fs_eval_dir" value={evalDir} onChange={e => setEvalDir(e.target.value)} />
      </FormRow>
      <FormRow label="WSR Test Output Directory:" htmlFor="fs_test_dir">
        <Input id="fs_test_dir" value={testDir} onChange={e => setTestDir(e.target.value)} />
      </FormRow>
      <FormRow label="WSR Checkpoint Directory:" htmlFor="fs_test_checkpoint_dir">
        <Input id="fs_test_checkpoint_dir" value={testCheckpointDir} onChange={e => setTestCheckpointDir(e.target.value)} />
      </FormRow>

      <SectionTitle>Deletion Flags</SectionTitle>
      <FormRow label="WandB Logs:">
        <Checkbox id="fs_keep_wandb" label="Keep Weights and Biases Logs (Uncheck to Delete)" checked={keepWandb} onChange={e => setKeepWandb(e.target.checked)} />
      </FormRow>
      <FormRow label="Train Logs:">
        <Checkbox id="fs_keep_log" label="Keep Train Logs (Uncheck to Delete)" checked={keepLog} onChange={e => setKeepLog(e.target.checked)} />
      </FormRow>
      <FormRow label="Model Weights:">
        <Checkbox id="fs_keep_output" label="Keep Model Weights (Uncheck to Delete)" checked={keepOutput} onChange={e => setKeepOutput(e.target.checked)} />
      </FormRow>
      <hr className="border-gray-700 my-4" />
      <FormRow label="Datasets:">
        <Checkbox id="fs_delete_data" label="Delete Generated Datasets" checked={deleteData} onChange={e => setDeleteData(e.target.checked)} />
      </FormRow>
      <FormRow label="Evaluation Results:">
        <Checkbox id="fs_delete_eval" label="Delete Evaluation Results" checked={deleteEval} onChange={e => setDeleteEval(e.target.checked)} />
      </FormRow>
      <FormRow label="Simulator Output:">
        <Checkbox id="fs_delete_test_sim" label="Delete WSR Simulator Output" checked={deleteTestSim} onChange={e => setDeleteTestSim(e.target.checked)} />
      </FormRow>
      <FormRow label="Simulator Checkpoints:">
        <Checkbox id="fs_delete_test_sim_checkpoint" label="Delete Simulator Checkpoints" checked={deleteTestSimCheckpoint} onChange={e => setDeleteTestSimCheckpoint(e.target.checked)} />
      </FormRow>
      <FormRow label="Cache Directories:">
        <Checkbox id="fs_delete_cache" label="Delete Cache Directories" checked={deleteCache} onChange={e => setDeleteCache(e.target.checked)} />
      </FormRow>
      
      <hr className="border-gray-700 my-4" />
      <FormRow label="Preview Mode:">
        <ToggleButton checked={preview} onClick={() => setPreview(c => !c)} variant="red-green">
          {preview ? "Preview Delete (Safe)" : "Perform Delete (Dangerous)"}
        </ToggleButton>
      </FormRow>
    </AppTab>
  );
};

export const FileSystemCryptographyTab: React.FC = () => {
  const [envFile, setEnvFile] = useState('vars.env');
  const [symkeyName, setSymkeyName] = useState('');
  const [cryptAction, setCryptAction] = useState('encrypt');
  const [targetFile, setTargetFile] = useState('');
  const [cryptPreview, setCryptPreview] = useState(true);

  return (
    <AppTab>
      <SectionTitle className="mt-0">Cryptography Settings</SectionTitle>
      <FormRow label="Environment Variables File:" htmlFor="fs_env_file">
        <Input id="fs_env_file" value={envFile} onChange={e => setEnvFile(e.target.value)} />
      </FormRow>
      <FormRow label="Symmetric Key Name:" htmlFor="fs_symkey_name">
        <Input id="fs_symkey_name" value={symkeyName} onChange={e => setSymkeyName(e.target.value)} placeholder="e.g., my_key" />
      </FormRow>
      <FormRow label="Action:">
        <div className="grid grid-cols-2 gap-2">
          <ToggleButton checked={cryptAction === 'encrypt'} onClick={() => setCryptAction('encrypt')} variant="blue">
            Encrypt
          </ToggleButton>
          <ToggleButton checked={cryptAction === 'decrypt'} onClick={() => setCryptAction('decrypt')} variant="blue">
            Decrypt
          </ToggleButton>
        </div>
      </FormRow>
      <FormRow label="Target File:" htmlFor="fs_target_file">
        <Input id="fs_target_file" value={targetFile} onChange={e => setTargetFile(e.target.value)} placeholder="path/to/file" />
      </FormRow>
      <FormRow label="Preview Mode:">
        <ToggleButton checked={cryptPreview} onClick={() => setCryptPreview(c => !c)} variant="red-green">
          {cryptPreview ? "Preview Crypt (Safe)" : "Perform Crypt (Dangerous)"}
        </ToggleButton>
      </FormRow>
    </AppTab>
  );
};

export const FileSystemToolsApp: React.FC = () => (
  <AppContainer
    tabs={[
      { name: 'Scripts', content: <FileSystemScriptsTab /> },
      { name: 'Update', content: <FileSystemUpdateTab /> },
      { name: 'Delete', content: <FileSystemDeleteTab /> },
      { name: 'Crypto', content: <FileSystemCryptographyTab /> },
    ]}
  />
);