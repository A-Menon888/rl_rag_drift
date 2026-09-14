from dataclasses import asdict
import json
from .generator import Query

def save_queries(queries: list[Query], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(query) for query in queries], handle, indent=2)
