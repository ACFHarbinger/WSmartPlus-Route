# Brief: paper update of 2026-09-26 (all agents, one file)

**Report of record:** `.agent/cache/paper_update_2026-09-26.md`. Read §0 and §1 first.
**Bus:** `.agent/bus/2026-09-26.md`.
**Paper:** `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/paper.tex`,
at upstream `399d22c`. The line numbers below refer to that commit.
**Beamer:** `temp/mpvrpp_beamer.pdf`, 35 slides, in Portuguese. It reproduces Ramos, de Morais &
Barbosa-Póvoa (2018), ESWA 103, and extends it to the MPVRPP; slide 33 marks what comes from the
source article and what is an extension. The paper is in English, so translate; do not paste
Portuguese.

## The ask (owner, 2026-09-26)

Every agent reads the beamer and proposes updates to the paper, using both the beamer and the
codebase:

- **Notation.** The paper's notation follows the beamer, except that the beamer's `S_i` stays `w_i`
  (the waste at bin `i`). The draft canonical table is report §1. Amend it in §1.b; do not invent a
  parallel table.
- **Algorithms.** Every description in Methodology must match what the code on `main` does.
- **Anything else** that improves the paper, one row each.

**Phase 1 is proposals only.**

- Write rows into the report. Each row has the exact old text, the new LaTeX, and its source (a
  beamer slide, a code `file:line`, a log or csv file).
- Do not edit `paper.tex`. Do not commit inside the submodule. Do not push anywhere.
- Once the owner rules, the accepted groups become GitHub issues and are delegated for
  implementation.

## Everyone reads

1. The whole beamer. Read the PDF pages, not only the text dump
   (`~/.cache/wsr-review/beamer.txt`, where formulas are garbled). Slides 21–25 hold the MPVRPP
   model and slides 32–34 hold the policy bridge and the limitations.
2. `paper.tex` at `399d22c`, in full.
3. §3 of `paper_versaoHector1.tex` (lines 359–920). It is a formulation draft in the old notation
   (`d`, `C_i`, `r_w`, `L_ij`, …) that overlaps with the beamer. Reconcile it; do not duplicate it.
4. The results in your lane from `.agent/cache/logic_review_2026-09-26.md`: the P rows describe
   algorithm/paper deviations, and the B rows may change reported numbers.

## Lanes

The lanes are the same agents as in the logic review. Lane boundaries decide who owns a row, not
what you may read.

| Agent | Paper sections owned | What to check against |
|---|---|---|
| **Kimi** (D) | §3 Problem Definition (lines 233–301); the exact constructors SWC-TCF and BPC (555–607); ACO-HH (715–748) | Beamer slides 4–27 and 31–33. Also `smart_waste_collection_two_commodity_flow/**`: does the code implement model 2, and in which reading of (16) and (18) (slide 17)? Propose the new §3 structure: description, assumptions, notation table, model 2 → MPVRPP extension with (22)–(44), terminal effect, and the three-stage bridge. Propose the route notation that replaces `𝒜_{d,k}`, `a_{t,k}` and `T_{d,k}`. |
| **Cursor** (F) | Mandatory Selection Strategies (468–543) | Beamer slides 10, 17–19 and 32: the fill rule `M`, the service level `δ`, the threshold `ψ`, the smarter trigger `H ≤ nδ`, and (40) as the mandatory set. Map LM, SL and LA to these precisely, with parameter values and units from the code (`threshold = 70.0` is a percent) and the configs actually used in the reported runs. |
| **Qwen** (E) | Meta-heuristics: ALNS, HGS, SANS, PG-CLNS, PSOMA (608–712) | The code on `main` plus your logic-review P rows. PG-CLNS is described as HVPL-inspired (owner, 2026-09-26). Give each algorithm the parameter values used in the paper's experiments (the yaml actually run). |
| **Gemini** (C) | The Neural Agent / Attention Model wherever the paper describes it; architecture figures under `Images/Architectures/`; training curves | `models/core/attention_model/**`, `neural_agent/**`, your P rows against Kool et al. Is the NA part of the reported experiments? If it is not, say so and propose what the paper should say. |
| **Grok** (B) | Simulation Protocol (779–847), Scenarios and Data (889–926), Data Integrity (1228–1260); the simulation-loop figure | The simulator code. The metric definitions must match the code: profit, kg/km, overflow = every day a bin sits full (DS-16), time = full policy time (DS-15), kg lost. Also the stochastic accumulation `a_i^t ∼ P_i` against the beamer's expected `a_i^t`, and the distance matrices (is anything symmetrised? Hector's remark (a)). |
| **Codex** (A + review) | Improvers CLS and Fast-TSP (749–778); Design of Experiments (927–963); every results table and number (964–1227, `Tables/*.tex`); **reviewer of all rows** | Every number traces to `docs/private/global/simulation/simulation_summary*.csv` or to a log under `assets/output/`. Fill report §4: which tables and claims are exposed to the bugs fixed since the runs (cf90 mirroring, DS-15, BPC, HGS split) or to the open logic-review B rows. |
| **Mistral** (G) | Abstract, Introduction, Related Work (note the duplicated `\section{Related Work}` at lines 226 and 302), Discussion, Conclusion (1261–1435); `mybibliography.bib`/`reference.bib` | A whole-document notation sweep: every math symbol in every section, table, caption and figure label against the §1 table, as a list of old → new occurrences with line numbers. Add the beamer's references (Ramos 2018, Baldacci 2004, Archetti 2014, Faccio 2011, Mes 2012) and the algorithm papers from `bibliography/`. Check the LaTeX build in a copy (below). |

DeepSeek, OpenCode and Muse may act as independent verifiers. Each writes its own §7 section,
re-checks rows against the beamer and the code, and flags disagreements.

## Rules

1. **Notation discipline.**
   - Use the report §1 symbols. If a symbol has an open owner question (Q2–Q5), use the proposed
     form, add `[Qn]` in the row, and move on.
   - Every row that touches math lists the symbols it introduces.
2. **Sources.**
   - A beamer claim cites its slide number.
   - A code claim cites `file:line` on `main` (re-check with `sed -n`).
   - A results claim cites the csv or log file.
   - Mark the beamer's own extensions (slide 33 "EXTENSÃO") as the authors' extensions in the
     proposed text. Do not attribute them to Ramos et al.
3. **Algorithm text describes the code that ran.** If the code and the paper differ, the row says
   which one is right and why. If a bug means the paper's description was right but the code was
   wrong, file it in report §4.
4. **LaTeX checks only in a copy.** Run:

   ```bash
   cp -r <paper> ~/.cache/wsr-review/<agent>-paper
   cd ~/.cache/wsr-review/<agent>-paper
   latexmk -pdf paper.tex
   ```

   LaTeX builds are light, but still do not run the simulator for this task. Rerunning experiments
   is Phase 2 work that the owner schedules.
5. **Resources.** The same rules as in the logic review apply (common brief "Resource rules").
   Heavy jobs need the `flock` lock and `sim.cpu_cores=2`.
6. **Row IDs.**
   - `N-<agent>-NN` for notation amendments (§1.b);
   - `F-<agent>-NN` for formulation (§2);
   - `A-<agent>-NN` for algorithms (§3);
   - `R-<agent>-NN` for results at risk (§4);
   - `I-<agent>-NN` for other improvements (§5).
7. **Write each row so it can become a GitHub issue:** one self-contained change, with an
   acceptance criterion.

## Done

Post `### <Agent> — 2026-09-26 (paper update lane <X> done)` on the bus. Give:

- the row counts per table;
- your three most important proposals;
- any owner question you want added to report §6.

Codex reviews the rows. Claude and Codex then consolidate report §8 into the GitHub issue list.
