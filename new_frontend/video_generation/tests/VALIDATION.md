# Validation of the portable sleeve repair

Validated locally against the completed Actor08 shorts + shirt experiments.
These checks reuse saved data and CPU SMPL-X; no training was repeated.

| Check | Result |
|---|---|
| Configuration / input-safety tests | 9 passed |
| Male 175 cm / 75 kg, restore-rebind vs. the validated v2 experiment | Passed; maximum `lbs.v` difference 1.103e-6 m |
| Female 175 cm / 75 kg, radial vs. the validated v1 experiment | Passed; maximum `lbs.v` difference 7.451e-7 m |
| `lbs_weights`, `posedirs`, zero `shapedirs` | Identical to the corresponding experiment in both modes |
| `rest_pos`, faces, node_type | Unchanged |
| Input, reference and expected files | SHA256 unchanged during regression |
| Saved male v2 `35_01` rollout, 90 trajectory frames | Wrist-patch proxy max: left 5.049 cm, right 4.409 cm |
| Python parsing and example JSON | Passed |

The small vertex differences are floating-point differences, below 0.002 mm.
The motion proxy is **not ground-truth fit error** and cannot certify absence
of penetration. Visual review of the actual motion is still required.

To repeat numerical regression with your own saved experiment:

```bash
"$CC_PY" tests/regression_sleeve_repair.py \
  --input /data/target_bound_before_repair.pkl \
  --reference reference_assets/Actor08_shirt_unrelaxed.pkl \
  --expected /data/validated_repair.pkl \
  --body-model-root /data/body_models \
  --gender male --mode restore-rebind \
  --report /output/new_regression_report.json
```

For female radial regression, use `--gender female --mode radial`; the
reference argument is optional in that mode. Model/reference data is not
included in Git. The private reference accompanies the local sharing bundle.
