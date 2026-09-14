from dataclasses import asdict, dataclass
from typing import Dict, List
import json
import random

ATTRIBUTES = ["CEO", "headquarters", "founded", "industry", "employees"]

@dataclass(frozen=True)
class Fact:
    entity: str
    attribute: str
    value: str
    valid_from: int
    valid_to: int | None
    version: int
    document_id: str

    @property
    def text(self) -> str:
        return f"{self.entity} {self.attribute} = {self.value}."

@dataclass(frozen=True)
class Query:
    query_id: str
    entity: str
    attribute: str
    text: str
    answer_by_version: Dict[int, str]
    direct_answer: str
    affected_by_drift: bool

    def answer(self, version: int) -> str:
        return self.answer_by_version[version]

def generate_database(seed: int = 7, entities: int = 40, facts_per_entity: int = 5) -> List[Fact]:
    rng = random.Random(seed)
    people = ["Alice", "Bruno", "Carla", "Daniel", "Elena", "Fatima", "Gavin", "Hana"]
    industries = ["software", "robotics", "biotech", "energy", "finance"]
    facts = []
    for i in range(entities):
        entity = f"{['Nova', 'Lumen', 'Orbit', 'Vertex', 'Cedar'][i % 5]}{i:02d}"
        values = {
            "CEO": people[i % len(people)],
            "headquarters": f"City-{i % 12}",
            "founded": str(1980 + i % 35),
            "industry": industries[i % len(industries)],
            "employees": str(50 + i * 37),
        }
        for attribute in ATTRIBUTES[:facts_per_entity]:
            facts.append(Fact(entity, attribute, values[attribute], 0, None, 0, f"doc-v0-{i:03d}-{attribute}"))
    return facts

def save_facts(facts: List[Fact], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(fact) for fact in facts], handle, indent=2)

def load_facts(path: str) -> List[Fact]:
    with open(path, encoding="utf-8") as handle:
        return [Fact(**item) for item in json.load(handle)]

def generate_queries(base_facts: List[Fact], count: int = 300, seed: int = 7) -> List[Query]:
    rng = random.Random(seed)
    candidates = [(f.entity, f.attribute) for f in base_facts]
    queries = []
    for i in range(count):
        entity, attribute = candidates[i % len(candidates)]
        fact = next(f for f in base_facts if f.entity == entity and f.attribute == attribute)
        wording = rng.choice([f"What is the {attribute} of {entity}?", f"Tell me the {attribute} for {entity}."])
        queries.append(Query(f"q-{i:04d}", entity, attribute, wording, {0: fact.value}, fact.value, False))
    return queries
