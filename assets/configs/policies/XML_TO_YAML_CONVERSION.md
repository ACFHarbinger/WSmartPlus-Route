# XML to YAML Conversion Summary

## Overview
All XML configuration files in `assets/configs/policies/` have been converted to YAML format with comprehensive documentation.

## Conversion Mapping

### Must-Go Strategy Files (7 files)

| Original XML | New YAML | Strategy | Key Parameters |
|--------------|----------|----------|----------------|
| `mg_lookahead.xml` | `mg_lookahead.yaml` | lookahead | current_collection_day: 0 |
| `mg_last_minute_cf70.xml` | `mg_last_minute_cf70.yaml` | last_minute | threshold: 0.7 |
| `mg_last_minute_cf90.xml` | `mg_last_minute_cf90.yaml` | last_minute | threshold: 0.9 |
| `mg_regular_lvl3.xml` | `mg_regular_lvl3.yaml` | regular | frequency: 3 |
| `mg_regular_lvl4.xml` | `mg_regular_lvl4.yaml` | regular | frequency: 4 |
| `mg_regular_lvl5.xml` | `mg_regular_lvl5.yaml` | regular | frequency: 5 |
| `mg_service_level0.84.xml` | `mg_service_level0.84.yaml` | service_level | confidence_factor: 0.84 |

### Post-Processing Files (3 files)

| Original XML | New YAML | Methods | Key Parameters |
|--------------|----------|---------|----------------|
| `pp_ftsp.xml` | `pp_ftsp.yaml` | fast_tsp | iterations: 1000 |
| `pp_cls.xml` | `pp_cls.yaml` | 2opt | ls_operator: 2opt, iterations: 1000 |
| `pp_rds.xml` | `pp_rds.yaml` | random_local_search | Operator weights, iterations: 1000 |

## Schema Alignment

All YAML files now align with the Python dataclass schemas:
- **Must-Go**: `logic/src/configs/must_go.py` → `MustGoConfig`
- **Post-Processing**: `logic/src/configs/post_processing.py` → `PostProcessingConfig`

## Policy File Updates

All 13 policy configuration files have been updated to reference `.yaml` instead of `.xml`:

1. `policy_alns.yaml`
2. `policy_bcp.yaml`
3. `policy_cvrp.yaml`
4. `policy_hgs.yaml`
5. `policy_hgs_alns.yaml`
6. `policy_hh_aco.yaml`
7. `policy_ks_aco.yaml`
8. `policy_lkh.yaml`
9. `policy_neural.yaml`
10. `policy_sans.yaml`
11. `policy_sisr.yaml`
12. `policy_tsp.yaml`
13. `policy_vrpp.yaml`

### Example Change
```yaml
# Before
- must_go: ["mg_lookahead.xml"]
- post_processing: []

# After
- must_go: ["mg_lookahead.yaml"]
- post_processing: []
```

## New YAML Features

Each converted YAML file now includes:
1. **Comprehensive header** explaining purpose, algorithm, and use cases
2. **Parameter documentation** with ranges and recommendations
3. **Performance expectations** for runtime and solution quality
4. **Schema compliance** with Python dataclass definitions

## Original XML Files

The original `.xml` files are preserved and can be found in the same directory. They may be removed after validating the YAML conversion works correctly with the codebase.

## Migration Checklist

- [x] Convert all XML files to YAML format
- [x] Add comprehensive documentation to each YAML file
- [x] Update all policy config files to reference `.yaml` extensions
- [x] Align YAML structure with Python dataclass schemas
- [ ] Test loading YAML files with `MustGoConfig` and `PostProcessingConfig`
- [ ] Update code loaders to support both `.xml` and `.yaml` (backward compatibility)
- [ ] Run integration tests with new YAML configs
- [ ] Remove `.xml` files after successful validation

## Notes

- All YAML files maintain the same parameter values as the original XML files
- Documentation was added to explain what each parameter does
- The conversion preserves backward compatibility by keeping original XML files
- File sizes increased due to comprehensive inline documentation
