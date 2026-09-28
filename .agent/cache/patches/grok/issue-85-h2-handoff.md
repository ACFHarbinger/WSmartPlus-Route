# Issue 85 / H2 handoff (Grok)

Base: paper branch `paper-update/beamer-notation` at `9898e2f`. The shared checkout was not edited.

Patches (apply from the repo that contains the path):

- `.agent/cache/patches/grok/issue-85-h2-protocol.patch` — `paper.tex` and `Images/Results/Generated/simulation_loop.png`, from the paper repo root. `git apply --check` passed on a copy of `9898e2f`.
- `.agent/cache/patches/grok/issue-85-h2-figure-generator.patch` — `logic/gen/gen_paper_latex.py` `fig_simulation_loop`, from the superproject root. `git apply --check` passed.

Build of the patched paper copy (`~/.cache/wsr-review/grok-h2`): `latexmk -pdf` exited 0, no undefined references. The only overfull boxes are pre-existing and outside this step: assumption (A4) at lines 567–569, and the Spinelli appendix at lines 2316–2319.

## What the patch changes

- **Day order.** Definition `def:mp-state` collects, then receives `a_{i,d}`. The simulator adds the day's draw first, caps at `Cap_i`, scores `o_{i,d}` and lost mass `\ell_{i,d}`, and only then routes. The reported indicator is the capped pre-collection level, not the model's test `w_{i,d} > Cap_i`.
- **Seed.** Given the scenario and the sample, the deposit of bin `i` on day `d` is fixed by `i` and `d`. The demand seed does not depend on the policy (`initializing.py` seeds waste with `sim.seed + sample_id` only). Levels are each policy's own residual.
- **Efficiency.** Defined once, before the results: total kilograms over total kilometers, not the mean of the daily ratios. That is `finishing.py` (sum of collected mass over sum of travel).
- **Q versus density.** `Cap_i` is `2.5 m³ × B`: 47.5 kg in Rio Maior (`B = 19`) and 50 kg in Figueira da Foz (`B = 20`). `Q` is the fleet payload, 3,500 kg and 2,500 kg, not computed from `B`.
- **Gamma-3.** Preset 2 in `logic/src/constants/data.py` (`GAMMA_PRESETS`), synthetic, not a fit to the municipal series. Unclipped `P(draw > 100%)` for pairs `(shape, scale)` `(1,8)`, `(1,6)`, `(3,8)`, `(3,6)` is `3.7e-6`, `5.8e-8`, `3.4e-4`, `9.0e-6`. The ten-bin tile (weights 3, 2, 2, 3) mixes to `7.2e-5`, about one bin-day in 14,000. Clipping to `[0, 100]` is what enforces assumption (A3) in the current generator. Whether the archived file used that clip is pending recovery (Q8).
- **Directed matrices versus (A1).** Runs price `dist_ij` in the direction driven and do not symmetrize. On `data/simulator/distance_matrix/submatrix/gmaps_distmat100_plastic[riomaior].csv`, 87% of ordered pairs differ by more than a meter (max gap 21.9 km), and 554 of 20,000 random directed triples violate the triangle inequality (2.8%).
- **City and size.** Figueira da Foz is the only `N = 350` network. The two Rio Maior focus graphs are subsets of the same 173-bin plastic fleet and share 97 bins; `N = 170` omits three bins.
- **Data Integrity** is moved to immediately before Experimental Results, and the text says the excluded runs are three at 30 days and one at 90 days. The forward reference in the selection section now says "above".
- **Figure.** `fig_simulation_loop` uses day `d`, horizon `D`, `Cap_i`, `a_{i,d}`, `o_{i,d}`, `M_d`, and `dist_ij`. The sensor panel stays gray. Efficiency in the log box is total kg / total km. The regenerated PNG is in the paper patch.

## D6, not printed as established

The owner ruling says `N = 100` is the 100 bins closest to the depot and `N = 170` the 170 farthest. Against Google depot distances, with focus-graph indices read as positions in the plastic rows of `old_out_info[riomaior].csv` (173 bins, IDs aligned with `gmaps_distmat_plastic[riomaior].csv`), that rule recovers **167 of 170** and only **54 of 100**. The paper states the description and this mismatch. It does not claim the published index lists are exactly those rankings.
