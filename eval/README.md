# Text2SQL Offline Evaluation

This directory contains a lightweight regression harness for the data-agent
LangGraph workflow. It is intended for prompt, retrieval, and graph changes
where we need to prove that the Text2SQL chain did not regress.

## Directory layout

    eval/
    ├── data/                    # Evaluation inputs and reviewed goldens
    ├── reports/
    │   ├── baseline/            # Full and golden replay baselines
    │   ├── smoke/               # Small stage-specific integration reports
    │   └── archive/             # Superseded historical reports
    ├── runner.py                # Runs the complete graph, including the LLM
    ├── metrics.py               # Result comparison and metric aggregation
    ├── llm_tracking.py          # LLM latency and token tracking
    ├── validate_goldens.py      # Verifies the reviewed answers
    └── replay_report.py         # Re-executes saved SQL without calling the LLM

## Run

```bash
python -m eval.runner
```

Run a small smoke test:

    python -m eval.runner --limit 5

Run only the 20 manually reviewed golden cases:

    python -m eval.runner --only-gold --report eval/reports/baseline/golden_20_report.json

Validate that the saved golden results still match the current DW data. This
does not call the LLM:

    python -m eval.validate_goldens

Replay SQL from the latest full report against the golden results, without
calling the LLM:

    python -m eval.replay_report

Use a custom case file and report path:

```bash
python -m eval.runner \
  --cases eval/data/questions.json \
  --report eval/reports/eval_report.json
```

## Case Format

`data/questions.json` is a JSON array. JSONL is also supported.

```json
{
  "id": "q001",
  "question": "统计各地区的销售总额",
  "expect_tables": ["fact_order", "dim_region"],
  "expect_not_empty": true,
  "difficulty": "easy",
  "category": "分组聚合",
  "expected_result": [
    {"region_name": "华东", "total_amount": 12345.0}
  ]
}
```

expected_result is optional. The manually reviewed answers live in
data/golden_results.json and are merged into cases by ID. When provided, the runner
compares complete result rows with:

- numeric absolute tolerance (default: 0.01);
- common metric aliases such as GMV, total_amount, and total_sales;
- order-insensitive comparison by default;
- order-sensitive comparison when expected_result_ordered is true;
- a structured result diff in the JSON report when comparison fails.

When no cases configure expected_result, the metric is reported as N/A instead
of 0%.

## Observability

Every LangGraph node emits a completion timing event. The evaluation report
stores:

- every node invocation, including retries and errors;
- the slowest node for each case;
- average, P50, P95, and maximum node latency;
- every chat-model call and the node that triggered it;
- model call latency, call count, errors, and token usage when reported;
- SQL cache hits, misses, and semantic-rule bypasses.

Prompt and response contents are not copied into the observability records.
Only sizes, timings, status, model name, and usage metadata are stored. Because
some retrieval nodes run in parallel, summed node or LLM time may be greater
than end-to-end wall-clock time. Current nodes use non-streaming model calls,
so full-response latency is recorded and first-token latency remains null.

## Schema Linking evaluation

Column, metric, and value recall now emit structured candidate events. Column
and metric candidates record whether they came from vector retrieval, exact
name/alias matching, or both, together with retrieval and rerank scores. Value
events only record the matched column and score; field values are not copied
into evaluation reports.

For the reviewed cases, expected tables, columns, metrics, and JOIN keys are
derived from `gold_sql` with sqlglot. Reports include table recall, raw and
reranked Column Recall@K, final column recall, Metric Recall@K, final metric
recall, JOIN-key coverage, source-level recall, missing items, and candidate
counts. Cases without a reviewed SQL still contribute to table recall through
their configured `expect_tables`.

## SQL safety boundary

Before database validation or execution, generated SQL is parsed into a
sqlglot AST. The audit layer enforces a single read-only query, rejects write
or control statements and dangerous functions, checks tables and columns
against the linked schema, rejects cross-schema access and SELECT *, and adds
or caps the outer LIMIT at 10000 rows. Audit failures are emitted with stable
error codes and can be sent to the SQL correction node for a targeted retry.
After the retry budget is exhausted the graph terminates without invoking the
database execution node.

## SQL repair convergence guard

Every corrected SQL is checked before it returns to the AST audit. The guard
normalizes SQL into a formatting-insensitive fingerprint, records every repair
attempt, stops identical or cyclic candidates, and compares AST-level business
structure with both the previous SQL and the original generated SQL. Changes to
aggregations, JOIN count, filters and their literals, GROUP BY, DISTINCT, set
operations, ordering, or an existing literal LIMIT are treated as semantic
drift. Parse failures are deliberately delegated to the AST audit so that a
targeted retry remains possible.

Offline reports store every guard decision and aggregate stopped cases by
`NO_CHANGE`, `REPAIR_CYCLE`, or `SEMANTIC_DRIFT`.

## Metrics

- SQL generated rate
- SQL executable rate
- Expected table hit rate
- Non-empty result pass rate
- Expected result match rate
- Expected/actual result diff for failed golden cases
- Self-repair case count
- Schema Linking table, column, metric, and JOIN-key recall
- Vector and exact-alias source-level column recall
- Vector and exact-alias source-level metric recall
- Raw, reranked, and final candidate counts
- Repair-guard stopped case count and stop reasons
- Average latency
- Node latency: average, P50, P95, maximum, and errors
- LLM latency, call count, token usage, and errors
- SQL cache hit/miss/bypass count
- Grouped metrics by difficulty and category
