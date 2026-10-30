# Assessment of an external assistant's verification report

| Claim | My check | Verdict |
|---|---|---|
| TLC Apr-Jun 2016 parquet has only PULocationID/DOLocationID (no lat/lon, no ID) | Same as our own profiling of 3 files (34,891,690 rows) | Agrees |
| taxi_data.h5 is (4368, 266); 4368 = 91 x 48 | Shape printed in authors' notebook cell 4; 91 x 48 is arithmetic | Shape confirmed; 30-min slot only PROBABLE (notebook cell 5 divides by 24) |
| Vicinity = 9 nearest stations | Authors' notebook cell 12: `(-arr).argsort()[:9]` on adj[2] (similarity weights, node itself included) | Confirmed |
| P = Q = 12 | Notebook shows 12-step input blocks only | P confirmed, Q still inferred |
| 12 x 18 = 216 | Authors' layout is 9 x (12+12) = 216, not 12 x 18 | Same number, different layout (A-25) |
| Notebook split: first half train, second half test (182 chunks, 180 used) | Cells 7-8 | Confirmed |
| demand = pick-ups, supply = drop-offs | Notebook cell 3 | Confirmed |
| 'the exact data behind the paper's own numbers' | Consistent with P13/R-notes, not proven | Unproven |
| Mirrors are the original files | No checksums, files not received here | Unverified (D10) |
| '56 passed, 1 skipped' on this repo | Our own runs show different counts for the commits we know | Not reproducible from here |
