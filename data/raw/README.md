# `data/raw/` — provenance, expected files, checksums

The five raw CSVs are **never committed** (`.gitignore` excludes `data/raw/*`
apart from this README and `.gitkeep`). This file is the tracked record of
*what* should be here and *how to verify it*.

## Expected files

| File | Rows | Cols | Bytes | md5 | sha256 (first 16) |
|---|---:|---:|---:|---|---|
| `train_transaction.csv` | 590,540 | 394 | 683,351,067 | `58b4038d8715f5e11007b826bef00ce7` | `3a5c83ab6b3cc13d` |
| `train_identity.csv` | 144,233 | 41 | 26,529,680 | `8487db5001c8bad139f3318d5d3db416` | `b63c725d8377be90` |
| `test_transaction.csv` | 506,691 | 393 | 613,194,934 | `7ea6862ef4e078efb309e19fa49178fd` | `2a8e51f1d335a860` |
| `test_identity.csv` | 141,907 | 41 | 25,797,161 | `54ae784303f82e5cadeb2d899b05c6a8` | `3e5978cb13ca5e72` |
| `sample_submission.csv` | 506,691 | 2 | 6,080,314 | `a4dece4fe5e7a398319009b94f7b34a5` | `50d7e0d6fcfc6e49` |

Verify with:

```bash
md5sum data/raw/*.csv
python -m src.data.schema_check     # also re-checks columns, row counts, join rate
```

## Provenance

Canonical source: the **IEEE-CIS Fraud Detection** Kaggle competition
(<https://www.kaggle.com/competitions/ieee-fraud-detection/data>), data
contributed by Vesta Corporation.

The project brief nominates a Google Drive folder as the delivery mechanism:
`https://drive.google.com/drive/folders/1IJOdTOgh3ltskru7LkoH0v-4u7YyW8If`
→ `ieee-fraud-detection/` → the five CSVs above.

**How this working copy was obtained (2026-08-21, cloud agent session).**
The Drive folder was located through the session's Google Drive connector and
the five files were confirmed present with byte sizes exactly matching the
table above. The connector, however, only exposes file content as a base64
blob inside a tool response, which is not a viable transport for a 683 MB
file, and the files are owner-only (no link sharing), so an unauthenticated
`curl` against `drive.google.com` returns a Google sign-in page rather than
the CSV. The sandbox has no Kaggle credentials either.

The files were therefore fetched over the sandbox's allowlisted network from a
public Kaggle **dataset mirror** of the same competition release
(`kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection`, a 123,856,947-byte
zip of the five original files), and validated against the Drive copies:

* all five byte sizes match the Drive originals exactly;
* row counts match the published competition figures (590,540 / 506,691 train
  and test transactions; 144,233 / 141,907 identity rows);
* column counts, the `id-01`-vs-`id_01` naming quirk, and the 24.42% train
  identity-join rate all match the documented release.

This is the real competition data, not a synthetic stand-in. The md5/sha256
values above were computed from this working copy and should be treated as the
reference for future sessions; if a future download disagrees, the download —
not this table — is the thing to investigate.

## Schema notes discovered on inspection (Phase 0)

1. **`test_identity.csv` uses hyphenated column names** (`id-01` … `id-38`)
   whereas `train_identity.csv` uses underscores (`id_01` … `id_38`). This is a
   property of the released files. `src.data.load.normalise_identity_columns`
   rewrites hyphens to underscores at load time so no downstream code has to
   special-case the split.
2. **The identity join is sparse and non-random.** Only **24.42%** of training
   transactions have a matching `identity` row. The join must be a left join
   from `transaction`; an inner join would discard 75.6% of the data along an
   axis that correlates with the target (see the `has_identity_record` feature
   in Phase 3, and its fraud-rate contrast in Phase 2).
3. `isFraud` is present in `train_transaction.csv` only, as expected. All model
   evaluation therefore happens inside the labelled training period via a
   chronological holdout — the competition test set carries no labels.
4. The `V*` block is `V1`…`V339` (339 columns), matching the plan.
