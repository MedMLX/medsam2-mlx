# Runner test data

The real-image tests use these paths by default; the environment variables
override them:

- `MEDMLX_TEST_CT`: `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`, the converted
  HCC-TACE-Seg HCC_063 contrast CT.
- `MEDMLX_TEST_XRAY`: `~/.cache/medmlx/test-data/chest_xray.jpg`, the normal PA
  chest radiograph from Wikimedia Commons.
- `MEDMLX_TEST_REFERENCES`: `work/real-reference`, containing the regenerated
  `runner_reference.npz`, `runner_run.npz`, and `reference_small.npz`.

Generate the references with the MedSAM2 runner recorder, passing a clean
checkout of the revision in `tests/reference_cases.py` via `--medsam2-source`
and the reference directory via `--output-dir`. Use the same CT and X-ray
when recording and testing. The generated arrays include image crops, so keep
the reference directory outside Git.

Generate the reduced-graph reference from the pinned source and CT with:

```sh
python -m tests.record_small_reference \
  --medsam2-source "$MEDSAM2_SOURCE_ROOT" \
  --output-dir "${MEDMLX_TEST_REFERENCES:-work/real-reference}"
```

The reduced-graph parity tests use the pinned upstream checkout at
`~/.cache/medmlx/test-data/medsam2-source` by default;
`MEDSAM2_SOURCE_ROOT` overrides that path.
