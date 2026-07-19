from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CATALOG_PATH = Path(__file__).parents[2] / "conf" / "meta_config.yaml"


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    normalized_text = text.casefold()
    return sorted({term for term in terms if term and term.casefold() in normalized_text})


def lexical_column_matches(query: str, keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """Match stable column names and aliases without an embedding or LLM call."""
    search_text = " ".join([query, *(keywords or [])])
    matches: list[dict[str, Any]] = []
    for table in _load_catalog().get("tables") or []:
        table_name = str(table.get("name") or "")
        for column in table.get("columns") or []:
            column_name = str(column.get("name") or "")
            terms = [column_name, *(str(alias) for alias in column.get("alias") or [])]
            matched = _matched_terms(search_text, terms)
            if matched:
                matches.append({
                    "id": f"{table_name}.{column_name}",
                    "table": table_name,
                    "column": column_name,
                    "matched_terms": matched,
                })
    return sorted(matches, key=lambda item: item["id"])


def lexical_metric_matches(query: str, keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """Match configured metric names and aliases as a deterministic recall channel."""
    search_text = " ".join([query, *(keywords or [])])
    matches: list[dict[str, Any]] = []
    for metric in _load_catalog().get("metrics") or []:
        metric_name = str(metric.get("name") or "")
        terms = [metric_name, *(str(alias) for alias in metric.get("alias") or [])]
        matched = _matched_terms(search_text, terms)
        if matched:
            matches.append({
                "id": metric_name,
                "matched_terms": matched,
            })
    return sorted(matches, key=lambda item: item["id"])
