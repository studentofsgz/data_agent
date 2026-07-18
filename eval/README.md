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

## SQL safety boundary

Before database validation or execution, generated SQL is parsed into a
sqlglot AST. The audit layer enforces a single read-only query, rejects write
or control statements and dangerous functions, checks tables and columns
against the linked schema, rejects cross-schema access and SELECT *, and adds
or caps the outer LIMIT at 10000 rows. Audit failures are emitted with stable
error codes and can be sent to the SQL correction node for a targeted retry.
After the retry budget is exhausted the graph terminates without invoking the
database execution node.

## Metrics

- SQL generated rate
- SQL executable rate
- Expected table hit rate
- Non-empty result pass rate
- Expected result match rate
- Expected/actual result diff for failed golden cases
- Self-repair case count
- Average latency
- Node latency: average, P50, P95, maximum, and errors
- LLM latency, call count, token usage, and errors
- SQL cache hit/miss/bypass count
- Grouped metrics by difficulty and category
