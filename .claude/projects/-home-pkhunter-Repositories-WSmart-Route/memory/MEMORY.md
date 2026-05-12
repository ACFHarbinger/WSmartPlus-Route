# Memory Index

- [BPC paper review and bug fixes](project_bpc_paper_review.md) — Five bugs fixed in BPC solver after BHV2000 audit; branching order, bounds threshold, RF pair selection, missing param, LCI arc gating
- [VRPP action-mask NaN fix](project_vrpp_nan_fix.md) — Mandatory-bin + zero-waste deadlock in VRPPEnv._get_action_mask; fix adds `mask |= pending_mandatory` to override waste filter for mandatory bins
