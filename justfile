# WSmart-Route Justfile

red := '\033[0;31m'
green := '\033[0;32m'
yellow := '\033[0;33m'
blue := '\033[0;34m'
purple := '\033[0;35m'
cyan := '\033[0;36m'
bold := '\033[1m'
reset := '\033[0m'

# Default variables (can be overridden: just train problem=wcvrp)

problem := "vrpp"
model := "am"
encoder := "gat"
decoder := "glimpse"
area := "riomaior"
batch_size := "64"
temporal_horizon := "0"
samples := "1"
seed := "42"
marker := "fast"
strategy := "greedy"
n_cores := "20"
hpo_policy := "bpc"
hpo_method := "nsgaii"
hpo_workers := "1"
hpo_selection := ""
hpo_acceptance := ""
hpo_improver := ""
hpo_trials := "10"
hpo_samples := "5"
hpo_policy_kw := "max_cg_iterations"
hpo_selection_kw := ""
hpo_acceptance_kw := ""
hpo_improver_kw := ""
policies := "aco_hh,alns,bpc,hgs,pg_clns,psoma,sans,swc_tcf"

#policies := "abc,abpc_hg,aco_hh,aco_ks,adp,ahvpl,aks,alns_ipo,alns,amphh,arco,bb,bc,bp,bpc,cf_rs,cgh,cp_sat,cvrp,de,es_mcl,es_mkl,es_mpl,esdp,fa,filo,ga,genius,gihh,gls,gp_hh,gp_mp_hh,hgs_adc,hgs_alns,hgs_rr,hgs,hmm_gd_hh,hms,hs,hulk,hvpl,ils_bd,ils_rvnd_sp,ils,kgls,ks,lb_vns,lb,lbbd,lca,lkh3,lrh,ma_dp,ma_im,ma_ts,ma,mhh,mp_aco,mp_ils,mp_pso,mp_sa,ph,phh,popmusic,pso,psoda,psoma,qde,rens,rfo,rl_ahvpl,rl_alns,rl_gd_hh,rl_hvpl,rts,sa,sans,sca,shh,sisr,slc,src,ss_hh,st_ef,swc_tcf,ts,tsp,vns,vpl"
# --- Setup & Environment ---

# Sync dependencies using uv
sync:
    uv sync --all-groups --all-extras

# Install dependencies via pip
install:
    uv pip install -r requirements.txt || uv pip install -e .

# Generate executable
package:
    uv run pyinstaller simulator.spec

# --- Primary Execution Commands (Hydra-based) ---
# Train a model with Hydra configs

# Usage: just train problem=wcvrp model=am
train problem=problem model=model encoder=encoder decoder=decoder batch_size=batch_size temporal_horizon=temporal_horizon:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🚀 STARTING HYDRA TRAINING SESSION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model:" "{{ model }} ({{ encoder }})"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Batch Size:" "{{ batch_size }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" && \
    uv run python main.py train \
        envs={{ problem }} \
        models={{ model }} \
        model.encoder.type={{ encoder }} \
        hpo.n_trials=0 \
        train.batch_size={{ batch_size }}

# Run model evaluation with Hydra configs

# Usage: just eval model_path=./weights/best.pt dataset=data/test.pkl problem=wcvrp strategy=greedy
eval model_path="" dataset="" problem=problem strategy=strategy:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "📊 STARTING MODEL EVALUATION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model Path:" "{{ model_path }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Dataset:" "{{ dataset }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Strategy:" "{{ strategy }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py eval \
        eval.policy.model.load_path={{ model_path }} \
        eval.datasets=[{{ dataset }}] \
        eval.problem={{ problem }} \
        eval.val_size={{ samples }} \
        eval.decoding.strategy={{ strategy }}

# Run simulator testing with Hydra configs

# Usage: just test-sim policies="vrpp,alns" area=riomaior
test-sim policies=policies area=area samples=samples n_cores=n_cores:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧪 STARTING SIMULATION TESTING"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Policies:" "{{ policies }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Area:" "{{ area }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Samples:" "{{ samples }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py test_sim \
        sim.policies=[{{ policies }}] \
        sim.n_samples={{ samples }} \
        sim.graph.area={{ area }} \
        sim.cpu_cores={{ n_cores }}

# Remove targeted simulation runs from output artefacts.
#
# Usage:
#   just clean-results results_dir=assets/output/30_days/riomaior_100 distribution=emp constructor=alns
#   just clean-results results_dir=assets/output/30_days/riomaior_100 improver=ftsp dry_run=true
#
# Parameters:
#   results_dir  — path to the output folder (required)
#   distribution — space-separated distribution tags  (emp gamma1 gamma2 gamma3)
#   constructor  — route constructor(s)               (alns hgs aco_hh bpc psoma …)
#   ms_strategy  — mandatory-selection strategy/ies   (lookahead last_minute regular)
#   improver     — route improver(s)                  (ftsp rls rds none …)
#   dry_run      — set to "true" to preview without deleting (default: false)
#   quiet        — set to "true" to suppress per-item output (default: false)

results_dir := "assets/output/30_days/riomaior_100"
distribution := ""
constructor := ""
ms_strategy := ""
improver := ""
dry_run := "false"
quiet := "false"

clean-results results_dir=results_dir distribution=distribution constructor=constructor ms_strategy=ms_strategy improver=improver dry_run=dry_run quiet=quiet:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING SIMULATION RESULTS"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Results Dir:" "{{ results_dir }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Distribution:" "{{ distribution }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Constructor:" "{{ constructor }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "MS Strategy:" "{{ ms_strategy }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Improver:" "{{ improver }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Dry Run:" "{{ dry_run }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python -m logic.src.cli.target_parser \
        --results-dir {{ results_dir }} \
        $([ -n "{{ distribution }}" ] && echo "--distribution {{ distribution }}" || true) \
        $([ -n "{{ constructor }}" ] && echo "--constructor {{ constructor }}" || true) \
        $([ -n "{{ ms_strategy }}" ] && echo "--ms-strategy {{ ms_strategy }}" || true) \
        $([ -n "{{ improver }}" ] && echo "--improver {{ improver }}" || true) \
        $([ "{{ dry_run }}" = "true" ] && echo "--dry-run" || true) \
        $([ "{{ quiet }}" = "true" ] && echo "--quiet" || true)

# Clean up Route Constructor configs and implementations

# Usage: just cleanup-route-constructors acronyms="alns,bpc"
cleanup-route-constructors acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING ROUTE CONSTRUCTORS"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_route_constructors.py "{{ acronyms }}"

# Clean up other Policy configs and implementations (mandatory selection, route improver, acceptance criteria, selection and construction)

# Usage: just cleanup-policy-others acronyms="bmc,regular"
cleanup-policy-others acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING OTHER POLICIES"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_policy_others.py "{{ acronyms }}"

# Clean up Environment (Envs) configs and implementations

# Usage: just cleanup-envs acronyms="vrpp"
cleanup-envs acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING ENVIRONMENTS"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_envs.py "{{ acronyms }}"

# Clean up Model configs and implementations

# Usage: just cleanup-models acronyms="am"
cleanup-models acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING MODELS"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_models.py "{{ acronyms }}"

# Clean up RL Training Algorithm configs and implementations

# Usage: just cleanup-rl-algorithms acronyms="ppo"
cleanup-rl-algorithms acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING RL ALGORITHMS"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_rl_algorithms.py "{{ acronyms }}"

# Clean up Route Constructor policies used for training models with Imitation Learning

# Usage: just cleanup-imitation-policies acronyms="aco"
cleanup-imitation-policies acronyms="":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧹 CLEANING IMITATION POLICIES"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acronyms:" "{{ acronyms }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python logic/src/utils/packages/remove_imitation_policies.py "{{ acronyms }}"

# Clean up HPO config files and implementations
cleanup-hpo:
    uv run python logic/src/utils/packages/remove_hpo.py

# Clean up Meta-RL config files and implementations
cleanup-meta:
    uv run python logic/src/utils/packages/remove_meta.py

# Clean up Evaluation config files and implementations
cleanup-eval:
    uv run python logic/src/utils/packages/remove_eval.py

# Clean up Callback implementations, keeping only those specified in keep_list
# Usage: just cleanup-callbacks keep_list="logging,checkpoint"
cleanup-callbacks keep_list="":
    uv run python logic/src/utils/packages/remove_callbacks.py "{{ keep_list }}"

# Clean up Tracking components and databases from the codebase
cleanup-tracking:
    uv run python logic/src/utils/packages/remove_tracking.py

# Clean up UI components from the codebase
cleanup-ui:
    uv run python logic/src/utils/packages/remove_ui.py

# Clean up Enums and the GlobalRegistry from the codebase
cleanup-enums:
    uv run python logic/src/utils/packages/remove_enums.py

# Clean up Datasets, Distributions, and Network Strategies from the codebase
# Usage: just cleanup-data datasets="baseline" distributions="uniform" network="euclidean"
cleanup-data datasets="" distributions="" network="":
    uv run python logic/src/utils/packages/remove_data.py --datasets "{{ datasets }}" --distributions "{{ distributions }}" --network "{{ network }}"

# Clean up Security and File System Utilities from the codebase
cleanup-security:
    uv run python logic/src/utils/packages/remove_security.py

# Usage: just hpo-sim policy=hgs trials=100 method=nsgaii
hpo-sim policy=hpo_policy trials=hpo_trials method=hpo_method workers=hpo_workers selection=hpo_selection acceptance=hpo_acceptance improver=hpo_improver policy_kw=hpo_policy_kw selection_kw=hpo_selection_kw acceptance_kw=hpo_acceptance_kw improver_kw=hpo_improver_kw area=area samples=hpo_samples:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🔍 STARTING SIMULATION POLICY HPO"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Policy:" "{{ policy }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Method:" "{{ method }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Trials:" "{{ trials }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Workers:" "{{ workers }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Selection:" "{{ selection }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acceptance:" "{{ acceptance }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Improver:" "{{ improver }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Policy KW:" "{{ policy_kw }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Selection KW:" "{{ selection_kw }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Acceptance KW:" "{{ acceptance_kw }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Improver KW:" "{{ improver_kw }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Area:" "{{ area }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py hpo_sim tracking=hpo_sim \
        hpo_sim.policy_name={{ policy }} \
        hpo_sim.n_trials={{ trials }} \
        hpo_sim.method={{ method }} \
        hpo_sim.num_workers={{ workers }} \
        hpo_sim.selection_name={{ selection }} \
        hpo_sim.acceptance_name={{ acceptance }} \
        hpo_sim.improver_name={{ improver }} \
        hpo_sim.policy_keywords={{ policy_kw }} \
        hpo_sim.selection_keywords={{ selection_kw }} \
        hpo_sim.acceptance_keywords={{ acceptance_kw }} \
        hpo_sim.improver_keywords={{ improver_kw }} \
        hpo_sim.graph.area={{ area }} \
        hpo_sim.graph.n_samples={{ samples }}

# Generate data with Hydra configs

# Usage: just gen-data wcvrp 100 emp
gen-data problem=problem:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "📁 GENERATING DATASET"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py gen_data \
        data.problem={{ problem }}

# Launch the GUI
gui:
    uv run python main.py gui

# Launch the dashboard
dashboard:
    uv run streamlit run logic/dashboard_entry.py

# --- Codebase Validation ---

# Pyrefly type checking for logic
pyrefly-logic:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🔍 STARTING PYREFLY TYPE CHECKING"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Target:" "Logic"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run pyrefly check logic/src --output-format min-text

# Pyrefly type checking for GUI
pyrefly-gui:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🔍 STARTING PYREFLY TYPE CHECKING"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Target:" "GUI"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run pyrefly check gui/src --output-format min-text

# Count lines of code and comments (use --group-by-dir N to aggregate by directory depth)
count-loc group_by="0":
    uv run python logic/src/utils/validation/count_loc.py logic/src --group-by-dir {{ group_by }}

# Tree view of lines of code and comments
tree-loc:
    uv run python logic/src/utils/validation/tree_loc.py logic/src

# Check docstring coverage
check-docs:
    uv run python logic/src/utils/docs/check_docstrings.py logic/src
    #uv run python logic/src/utils/docs/check_docstrings.py gui/src

# Check Google style docstrings
check-google-docs:
    uv run python logic/src/utils/docs/check_google_style.py logic/src \
        --exclude_dir logic/src/pipeline/simulations/wsmart_bin_analysis/test/

# Check for multiple classes in one file
check-multi-classes:
    uv run python logic/src/utils/validation/check_multi_classes.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test

# Find all relative imports (from .module import ...) with optional stats or exclude_same_package=true
check-relative-imports exclude_same_package="":
    uv run python logic/src/utils/validation/check_relative_imports.py logic/src \
        --exclude pipeline/simulations/wsmart_bin_analysis/test --stats \
        {{ if exclude_same_package != "" { "--exclude-same-package" } else { "" } }}

# Check for nested imports (add --stats for a per-package summary table)
check-nested-imports:
    uv run python logic/src/utils/validation/check_nested_imports.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test --ignore_factories

# Check for nested imports with a per-package summary table
check-nested-imports-stats:
    uv run python logic/src/utils/validation/check_nested_imports.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test --ignore_factories --stats

# Detect circular import chains using Tarjan's SCC algorithm
check-circular-imports html="":
    uv run python logic/src/utils/validation/check_circular_imports.py logic/src \
        --exclude pipeline/simulations/wsmart_bin_analysis/test \
        {{ if html != "" { "--html " + html } else { "" } }}

# Check that all concrete classes fully implement their ABC/Protocol interface contracts
check-interface-compliance:
    uv run python logic/src/utils/validation/check_interface_compliance.py logic/src

# Measure type annotation coverage per file (worst-coverage files shown first)
check-type-coverage sort="coverage" limit="40":
    uv run python logic/src/utils/validation/check_type_coverage.py logic/src \
        --sort {{ sort }} --limit {{ limit }}

# Generate interactive module-level import graph (opens in browser)
# Scans from repo root so logic/gui layers are correctly distinguished.

# Use depth=N to collapse to top-N package levels; set html= to change output path.
module-graph html="module_graph.html" depth="10":
    uv run python logic/src/utils/validation/visualize_module_graph.py ./logic/src \
        --exclude .venv venv node_modules \
        --html {{ html }} --depth {{ depth }}

# Create a graph with exported and imported dependencies (function, classes, etc.)
dependency-graph target_file="logic/src/utils/helpers/wrappers.py" target_name="greedy_day_route":
    uv run python logic/src/utils/validation/trace_dependencies.py logic/src {{ target_file }} {{ target_name }}

# Check for embedded languages in the Python source code
check-embedded-languages:
    uv run python logic/src/utils/validation/check_embedded_languages.py logic/src

# Check for unused imports in the Python source code
check-unused-imports:
    uv run python logic/src/utils/validation/check_unused_imports.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test --ignore_factories

# --- Advanced Testing & Benchmarks ---

# Run mutation tests using mutmut
mutation-test:
    export PYTHONPATH=. && uv run mutmut run --paths-to-mutate logic/src/

# Show mutation test results
mutation-report:
    uv run mutmut show all

# Run performance benchmarks
benchmark:
    export PYTHONPATH=. && uv run python -m logic.benchmark.run_all

# --- Script Runners ---

# Run training script (scripts/train.sh)
run-train:
    bash scripts/train.sh

# Run evaluation script (scripts/evaluation.sh)
run-eval:
    bash scripts/evaluation.sh

# Run data generation script (scripts/gen_data.sh)
run-gen-data:
    bash scripts/gen_data.sh

# Run meta-training script (scripts/meta_train.sh)
run-meta:
    bash scripts/meta_train.sh

# Run HPO script (scripts/hyperparam_optim.sh)
run-hpo:
    bash scripts/hyperparam_optim.sh

# Run simulation script (scripts/test_sim.sh)
run-sim:
    bash scripts/test_sim.sh

# --- Test & Quality ---

# Run all tests
test:
    uv run pytest --cov=logic/src --cov=gui/src --cov-report=xml --cov-report=term-missing

# Run fast unit tests
test-fast:
    uv run pytest -m "fast or unit"

# Run logic tests
test-logic:
    uv run pytest logic/test/

# Run GUI tests
test-gui:
    uv run pytest gui/test/

# Run tests with a specific marker
test-marker marker=marker:
    uv run pytest -m "{{ marker }}"

# Check code quality with ruff
lint:
    uv run ruff check . --fix --exclude ".venv"

build-docs:
    cd logic/docs && uv run make html

# Format code with black and ruff
format:
    uv run ruff format . --exclude ".venv"

# Build the dashboard docker image
docker-build:
    docker build -t wsmart-dashboard .

# Run the dashboard in a container with live log mounting
docker-dashboard:
    docker run -p 8501:8501 \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/assets/output:/app/assets/output \
      wsmart-dashboard

# --- Maintenance ---

# Clean caches and artifacts
clean:
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".hypothesis" -exec rm -rf {} +
    find . -type f -name "coverage.json" -exec rm {} +
    find . -type f -name "coverage.xml" -exec rm {} +
    find . -type f -name ".coverage" -exec rm {} +
    rm -rf *.egg-info
    rm -rf logs/
    rm -rf dist/
    rm -rf temp/
    rm -rf build/
    rm -rf wandb/
    rm -rf mlruns/
    rm -rf scratch/
    rm -rf outputs/
    rm -rf checkpoints/
    rm -rf model_weights/
    # Remove all empty directories recursively
    find . -type d -empty -delete

clean-outputs:
    rm -rf assets/output/
    rm -rf assets/tracking/

# --- Tracking Database ---

# Print an overview of experiments, runs, and record counts
db-inspect:
    uv run python -m logic.src.tracking.database inspect

# Delete all tracking data while preserving the schema
db-clean:
    uv run python -m logic.src.tracking.database clean

# Integrity check, WAL checkpoint, and VACUUM
db-compact:
    uv run python -m logic.src.tracking.database compact

# Remove stale runs  (override: just db-prune older_than=7 status=failed experiment=AM-VRPP-50)
db-prune older_than="30" status="failed" experiment="" dry_run="":
    uv run python -m logic.src.tracking.database prune \
        --older-than {{ older_than }} \
        --status {{ status }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }} \
        {{ if dry_run != "" { "--dry-run" } else { "" } }}

# Export a run to JSON  (override: just db-export run_id=a1b2c3d4 output=run.json)
db-export run_id="" experiment="" output="" latest="":
    uv run python -m logic.src.tracking.database export \
        {{ if run_id != "" { "--run-id " + run_id } else { "" } }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }} \
        {{ if output != "" { "--output " + output } else { "" } }} \
        {{ if latest != "" { "--latest" } else { "" } }}

# Show comprehensive database statistics  (override: just db-stats experiment=AM-VRPP-50)
db-stats experiment="":
    uv run python -m logic.src.tracking.database stats \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }}

# Show per-metric statistics  (override: just db-metrics key=train/cost experiment=AM-VRPP-50)
db-metrics key="" experiment="":
    uv run python -m logic.src.tracking.database metrics \
        {{ if key != "" { "--key " + key } else { "" } }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }}

# Generic run command
run *args:
    uv run python main.py {{ args }}
