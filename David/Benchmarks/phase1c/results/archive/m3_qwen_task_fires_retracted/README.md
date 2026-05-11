# §1c.33 retracted artifacts

These two JSON dumps from the first pass of §1c.33 are preserved
for the record but should not be used to draw conclusions.

| File | Origin | Status |
|---|---|---|
| `eager_pre_reset.json` | First §1c.33 eager dump | Valid (eager doesn't capture, so no contamination), but kept here alongside the M3 pair for parallelism. |
| `m3_pre_reset_capture_contaminated.json` | First §1c.33 M3 capture dump | **RETRACTED.** Counts include graph-capture warmup fires; the `1.69× more fires per slab` reading in the original §1c.33 was a measurement artifact. |

The canonical reset-isolated artifacts live one directory up at
`m3_qwen_task_fires/{eager_reset.json, m3_reset.json}`. Those
were captured with `VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1`
so the dumped counters cover ONLY the measured generate's
replay window. See `David/Docs/phase1c_findings.md` §1c.33 +
§1c.33 review-fix for the full story.
