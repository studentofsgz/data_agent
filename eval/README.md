# Text2SQL Offline Evaluation

This directory contains a lightweight regression harness for the data-agent
LangGraph workflow. It is intended for prompt, retrieval, and graph changes
where we need to prove that the Text2SQL chain did not regress.

## Run

```bash
python -m eval.runner
```

Run a small smoke test:

```bash
python -m eval.runner --limit 5
```

Use a custom case file and report path:

```bash
python -m eval.runner --cases eval/questions.json --report eval/eval_report.json
```

## Case Format

`questions.json` is a JSON array. JSONL is also supported.

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

`expected_result` is optional. When provided, the runner compares the full
result rows after light normalization.

## Metrics

- SQL generated rate
- SQL executable rate
- Expected table hit rate
- Non-empty result pass rate
- Expected result match rate
- Self-repair case count
- Average latency
- Grouped metrics by difficulty and category
