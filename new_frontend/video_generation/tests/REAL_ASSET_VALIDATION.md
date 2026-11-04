# Real-asset validation — 2026-11-05

**Result: anatomical repair remains experimental. Do not replace the validated
Actor08-specific repair with it yet.** Passing the synthetic unit tests did not
establish real-garment quality.

## Setup

- Host: user's cvlab-111 account, isolated `garment_lbs_validation_20261105_*` directories.
- CPU only, `CUDA_VISIBLE_DEVICES=''`, two OMP/OpenBLAS threads. Existing GPU jobs were not changed.
- Target: existing fitted male 180 cm / 80 kg betas; original fixed-size prepare entry.
- Actor08 shirt: template frame 183, 7,724 vertices; authenticated same-topology unrelaxed reference.
- Actor05 trousers: template frame 1960, 7,985 vertices; source cache used **only as a healthy/no-op control**, not claimed to be an authenticated unrelaxed recovery reference.
- Four source garment/source-fit files retained identical SHA256 hashes.

## Shipped prototype (`766a423`) results

| Test | Result |
|---|---|
| Shirt restoration candidate | Rejected by geometric gates; no repaired dict published |
| Trousers healthy/no-op control | Passed; no region triggered restoration; geometry/binding unchanged |
| Shirt maximum reference-relative edge strain | Input 4.878 → candidate 1.073; still above 0.35 limit |
| Shirt 99th-percentile reference-relative edge strain | Input 0.957 → candidate 0.391 |
| Shirt reference-normal reversal proxy count | Input 536 → candidate 22, including 9 new flags |
| Synthetic/configuration tests | 25 passed |

Edge strain is `abs(candidate_edge_length / reference_edge_length - 1)`.
Values above are ratios, not distances. The maximum and P99 use edges touching
the moved region; edges shorter than 1e-7 m in the reference are excluded.

The field called `flipped_triangles` in the prototype is a **conservative normal
dot-product proxy**, not proof of a topological inversion in 3D: a large local
surface rotation can also reverse its normal relative to the reference. Its
counts must not be reported as confirmed inverted triangles. This limitation
also applies to the current gate and is one reason to retain experimental status.

## Diagnostic transition trials

Two isolated alternatives used bone-axis position for the proximal blend,
then additionally extended the blend through mixed-weight neighboring regions.
Both kept the same gates. Neither passed on the real shirt:

| Candidate | Maximum edge strain | Normal reversal proxy count | Accepted |
|---|---:|---:|---|
| Original anatomical prototype | 1.073 | 22 | No |
| Bone-axis proximal blend | 1.447 | 14 | No |
| Bone-axis blend + mixed-region transition | 1.447 | 22 | No |

The alternatives are retained in the private diagnostic artifacts, not shipped
as improvements. The branch keeps the original prototype algorithm; no gates
were loosened just to obtain a passing result.

## Simulation and remaining work

A 20-step CPU ContourCraft rollout was attempted on the accepted trousers
control. The existing checkpoint loader moved the model to CUDA before the
entry's `--device cpu` handling, and failed with `No CUDA GPUs are available`.
**No rollout or GG video was produced.** Both 110 and 111 had other users' GPU
processes at the time; no GPU simulation was launched. WSL returned
`Wsl/E_ACCESSDENIED`.

The anatomical shirt candidate should stay rejected until its transition is
improved and evaluated with reliable geometry/penetration checks. A subsequent
GPU rollout must cover initial and extreme poses; geometry-only and no-op
checks are insufficient. Keep using the previously validated Actor08-specific
mode for production videos.
