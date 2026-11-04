# Real-asset validation — final WSL revision, 2026-11-05

**PASS for the cases and gates below. This supersedes the rejected v1 prototype, not a guarantee for arbitrary garments.** No Stage 1/3, rest mesh, topology, pins or material parameters were edited; no source garment cache was overwritten. Input-file hashes were checked after testing. No packages were installed for this validation.

## What failed and what changed

1. Hard region replacement produced seam strain. The replacement now uses a topology-based harmonic residual, bounded support expansion, and edge/normal constraints. The 0.35 strain gate was retained.
2. Spatial deviation conflated collapsed sleeves with valid lateral offsets. The recommended axial trigger restores a genuinely shortened region; if only lateral deviation is large, input-preserving fit retains its folds/length. Healthy controls remain unchanged. There is no hardcoded male/female repair switch.
3. Ring-wise translations center the covered limb segment without resizing rings. Explicit long-sleeve/trouser profiles can align the tube end along the axis; shorts use the covered proximal segment and are not lengthened to the ankle.
4. The SMPL-X body has 32 open facial boundary edges. Raw signed-distance queries misclassified some chest points. A separate query-only copy caps that small facial loop, checks edge closure and uses five-ray voting; source models and the simulation collider remain unchanged. Other large/non-facial holes are rejected. Open3D requires watertight inputs for meaningful signs: https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html
5. Clearance and edge constraints alternate until both pass or the bounded solver rejects the candidate. Narrow-gap escape is bounded by the same displacement gate. No failing output is published.

## Environment and protocol

- Local WSL, existing `contourcraft-fixed-size-sm120` and GG environments, CUDA GPU.
- Actor08 shirt tf183, male/female 175 cm / 75 kg, their own target betas, common same-topology unrelaxed reference.
- Recommended profile: axial trigger 0.10; centering 0.30 of covered bone segment; clearance 0.006 m (0.0001 m numerical tolerance); max relative edge change 0.35; max support 0.80; explicit long-end alignment. Upper, shorts and long-trouser profiles are included.
- **33 tests passed**, covering anatomy, transform invariance, no-op controls, rejection, source preservation, topology transition, short coverage, contact gaps and query-only caps, plus configuration integration.
- Geometry candidates have zero reference-normal reversal flags. This is a conservative proxy, not proof of no self-intersection or physical validity.
- Male shirt strain 0.349517 uses unrelaxed shape reference; female 0.349587 uses input-preserving shape reference. These are different references and should not be interpreted as directly comparable distortion scores.

## Complete tested trajectories and actual GG samples

The table uses the complete trajectories returned by the existing motion loader, not the original NPZ frame count. `35_01` produced 90 trajectory frames, `137_26_t05` 160. A tracked cuff-candidate centroid is compared with a posed wrist-surface-patch centroid on every frame; this is a drift proxy, **not ground-truth error**. The 15 cm acceptance threshold is specific to these long sleeves.

| Body | Motion | Frames | Maximum left cuff distance (cm) | Maximum right cuff distance (cm) | Gate |
|---|---|---:|---:|---:|---|
| male 175/75 | 35_01 | 90 | 5.82 | 6.72 | PASS |
| male 175/75 | 137_26_t05 | 160 | 5.86 | 10.65 | PASS |
| female 175/75 | 35_01 | 90 | 7.72 | 7.90 | PASS |
| female 175/75 | 137_26_t05 | 160 | 10.43 | 8.37 | PASS |

GG samples were reviewed at initialization, worst cuff displacement, extreme arm bends/raises, middle and final frames. Both sleeves remain on the forearms after settling; the prior sleeve retreat to the elbow was not observed in these runs. Initialization frames still show Gaussian blur. Later renders do not establish zero penetration or perfect fabric appearance.

The final full tests disabled backward tracking **only inside the neural network forward**, keeping ContourCraft contour-energy/collision gradients active. Global no-grad was rejected because it breaks contour gradients. An unmodified gradient-enabled male 20-step run also passed. Independent trajectories are stochastic; bitwise equivalence between these execution paths was not established, and the validation helper is not installed into production overlays.

Canonical independent check: male vertices more than 2 mm inside the body decreased 16 → 0; female 30 → 6. Both repaired cuff subsets have zero such vertices. Female residual points are outside the repaired support; this is **not** a global zero-penetration result.

The last covered-segment extension for shorts was checked against the full-test snapshot on all four real shirt anchor arrays: **exactly zero difference**. Its new proximal-segment case has its own unit test; the long-sleeve render evidence therefore remains applicable.

## Other garments: healthy controls and controlled damage

| Garment | Healthy geometry/weights unchanged | Controlled 18 cm left-end collapse | Max reference-relative edge change |
|---|---|---|---:|
| Actor08 shorts | PASS | PASS | 0.349557 |
| Actor05 trousers tf1960 | PASS | PASS | 0.349581 |

These use real meshes with explicitly synthetic damage. They demonstrate a shared geometric repair rule, not a naturally observed trousers failure or full dynamic validation of repaired trousers. The Actor05 source cache is a known healthy control/reference for this stress test, not an authenticated pre-untangle recovery reference. Existing healthy shorts were included in the combined shirt motion tests, which does not turn the artificial shorts-stress test into a rollout validation.

## Reproduce and review

1. Follow `GARMENT_LBS.md`, use a same-topology canonical reference and a separate output directory. Do not overwrite an existing reference/cache.
2. Run unit tests using the existing CC Python: `python -m unittest discover -s tests -p 'test_*.py'`.
3. Run generic CLI diagnosis then `--apply` using the appropriate supplied profile. Run the normal combined-outfit pipeline in a new work/library root; inspect the JSON and canonical preview before full rollout.
4. Validate a short trajectory, then the entire action including extreme poses; retain sidecars and compare against the unchanged baseline.

Portable JSON metrics and actual GG sample sheets are in the sharing ZIP under `validation/`. The local raw run is `/home/user/garment_lbs_validation/run_20261105_033159`; its `final_status.json` records completed manual tests that supersede an intentionally aborted slow automatic 20-step test. Original failure/abort logs are retained. The physical test helper is environment-specific and is not a new production inference method.

Torso-only repairs, skirts, arbitrary references, other betas and severely disconnected garments remain unverified. A profile is opt-in per garment. No Git push was performed.

---

## Archived v1 validation (superseded; retained for traceability)

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
