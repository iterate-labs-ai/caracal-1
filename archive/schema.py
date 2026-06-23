"""Schema do arquivo evolutivo Darwin Godel + MAP-Elites."""
from __future__ import annotations
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class CheckpointScores(BaseModel):
    cybergym_pass_at_5: float
    cybergym_pass_at_1: float
    coverage_fraction: float
    cost_usd_per_eval: float
    anti_hack_correlation: float
    humaneval_plus: float


class Checkpoint(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    parent_ids: list[UUID] = Field(default_factory=list)
    lineage_depth: int = 0
    weights_uri: str  # s3://... ou huggingface://...
    config_hash: str  # SHA256 do config
    scores: CheckpointScores
    behavior_cell_id: str  # CWE x primitive x lang
    pareto_flag: bool = False
    children_with_edit_cap: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    promoted: bool = False
    promotion_record_id: UUID | None = None


class BehaviorCell(BaseModel):
    """MAP-Elites cell · 30 CWE x 5 primitives x 3 langs = 450 cells."""
    cwe_class: str
    exploit_primitive: str
    language: str

    @property
    def id(self) -> str:
        return f"{self.cwe_class}_{self.exploit_primitive}_{self.language}"


class HarnessConfig(BaseModel):
    """Mutavel via Loop D scaffold mutation."""
    prompt_scaffold_id: str
    tool_budget: int
    retrieval_depth: int
    state_machine_id: str
    temperature_schedule: list[float]


class TripletEntry(BaseModel):
    """Entrada do archive: checkpoint + harness + primitives."""
    checkpoint: Checkpoint
    harness_config: HarnessConfig
    primitive_library_version: str  # vetorial Qdrant snapshot id


CWE_CLASSES = [
    "CWE-119", "CWE-120", "CWE-125", "CWE-787",  # buffer overflows
    "CWE-415", "CWE-416",  # double-free, UAF
    "CWE-190", "CWE-191",  # integer over/underflow
    "CWE-476",  # null deref
    "CWE-369",  # divide by zero
    "CWE-787",  # OOB write
    "CWE-22",  # path traversal
    "CWE-78", "CWE-77",  # command/code injection
    "CWE-89", "CWE-94",  # SQL/code injection
    "CWE-79", "CWE-80",  # XSS
    "CWE-352",  # CSRF
    "CWE-269",  # priv esc
    "CWE-200", "CWE-209",  # info disclosure
    "CWE-362", "CWE-367",  # race conditions, TOCTOU
    "CWE-674",  # uncontrolled recursion
    "CWE-704",  # type confusion
    "CWE-787",  # uninit memory
    "CWE-704",  # format string
    "CWE-79",  # serialization
]

EXPLOIT_PRIMITIVES = [
    "memory_corruption",
    "logic_flaw",
    "race_condition",
    "type_confusion",
    "side_channel",
]

LANGUAGES = ["c_cpp", "python", "rust"]
