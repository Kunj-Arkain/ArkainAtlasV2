"""
engine.evidence — Evidence Object Model (#2)
==============================================
Every number in the engine must trace to a source.

EvidenceRef: immutable record of where a data point came from
EvidenceStore: file-backed store for all evidence collected during a run
EvidenceChain: links a variable to its evidence refs

No variable can be treated as "known" without at least one EvidenceRef.
"""

from __future__ import annotations
import hashlib
import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


class EvidenceMethod(Enum):
    """How the evidence was obtained."""
    API_CALL = "api_call"          # Tool/API returned data
    DOCUMENT_EXTRACT = "document"  # Parsed from uploaded doc
    USER_ENTRY = "user_entry"      # Manually entered by user
    CALCULATION = "calculation"    # Derived from other evidence
    EXTERNAL_DB = "external_db"    # External database lookup
    WEB_SCRAPE = "web_scrape"      # Web data extraction
    MODEL_OUTPUT = "model_output"  # LLM/ML model output (lower trust)


class EvidenceGrade(Enum):
    """Reliability grade — drives trust scoring."""
    A = "A"  # Primary source, verified (county records, SEC filings)
    B = "B"  # Reputable secondary (CoStar, Census API, BLS)
    C = "C"  # Estimated/modeled (comps analysis, ML prediction)
    D = "D"  # User assertion without backup
    F = "F"  # Unverified / single-source / stale


@dataclass(frozen=True)
class EvidenceRef:
    """Immutable evidence reference — the atomic unit of grounding.

    Every data point that enters the engine gets one of these.
    frozen=True ensures evidence can't be silently mutated.
    """
    ref_id: str                              # Unique ID: "ev_{hash[:12]}"
    source_name: str                         # "census_demographics", "user:jsmith", etc.
    source_url: Optional[str] = None         # URL or file path if applicable
    method: EvidenceMethod = EvidenceMethod.API_CALL
    grade: EvidenceGrade = EvidenceGrade.C
    retrieved_at: float = 0.0                # Unix timestamp
    payload_hash: str = ""                   # SHA-256 of raw payload
    raw_payload: Optional[str] = None        # The actual data (truncated)
    variable: str = ""                       # Which variable this evidences
    value: Any = None                        # The extracted value
    confidence: float = 0.5                  # Source confidence in this value
    stale_after_hours: float = 168.0         # Default: 1 week
    notes: str = ""

    def is_stale(self) -> bool:
        age_hours = (time.time() - self.retrieved_at) / 3600
        return age_hours > self.stale_after_hours

    def trust_score(self) -> float:
        """Composite trust: grade × freshness × method."""
        grade_scores = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "F": 0.15}
        method_scores = {
            EvidenceMethod.API_CALL: 0.9, EvidenceMethod.DOCUMENT_EXTRACT: 0.85,
            EvidenceMethod.EXTERNAL_DB: 0.9, EvidenceMethod.USER_ENTRY: 0.7,
            EvidenceMethod.WEB_SCRAPE: 0.6, EvidenceMethod.CALCULATION: 0.75,
            EvidenceMethod.MODEL_OUTPUT: 0.5,
        }
        base = grade_scores.get(self.grade.value, 0.5)
        method_mult = method_scores.get(self.method, 0.5)
        freshness = 0.5 if self.is_stale() else 1.0
        return round(base * method_mult * freshness, 3)

    def to_dict(self) -> Dict:
        d = {
            "ref_id": self.ref_id, "source_name": self.source_name,
            "method": self.method.value, "grade": self.grade.value,
            "retrieved_at": self.retrieved_at, "payload_hash": self.payload_hash,
            "variable": self.variable, "value": self.value,
            "confidence": self.confidence, "trust_score": self.trust_score(),
            "is_stale": self.is_stale(),
        }
        if self.source_url:
            d["source_url"] = self.source_url
        if self.notes:
            d["notes"] = self.notes
        return d


def make_evidence_ref(source_name: str, variable: str, value: Any,
                      method: EvidenceMethod = EvidenceMethod.API_CALL,
                      grade: EvidenceGrade = EvidenceGrade.C,
                      confidence: float = 0.5,
                      raw_payload: str = None,
                      source_url: str = None,
                      notes: str = "") -> EvidenceRef:
    """Factory function to create EvidenceRef with auto-generated ID + hash."""
    ts = time.time()
    payload_str = raw_payload or json.dumps({"value": value, "source": source_name})
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    ref_id = f"ev_{payload_hash[:12]}"
    return EvidenceRef(
        ref_id=ref_id, source_name=source_name, source_url=source_url,
        method=method, grade=grade, retrieved_at=ts,
        payload_hash=payload_hash,
        raw_payload=payload_str[:2000] if raw_payload else None,
        variable=variable, value=value, confidence=confidence,
        notes=notes,
    )


class EvidenceStore:
    """File/memory-backed store for all evidence in a run.

    Stores evidence refs, supports lookup by variable, source, or ID.
    Can persist to JSON for audit trail.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._refs: Dict[str, EvidenceRef] = {}          # ref_id → EvidenceRef
        self._by_variable: Dict[str, List[str]] = {}     # variable → [ref_ids]
        self._by_source: Dict[str, List[str]] = {}       # source → [ref_ids]
        self._persist_path = persist_path

    def add(self, ref: EvidenceRef) -> str:
        """Store an evidence ref. Returns ref_id."""
        self._refs[ref.ref_id] = ref
        self._by_variable.setdefault(ref.variable, []).append(ref.ref_id)
        self._by_source.setdefault(ref.source_name, []).append(ref.ref_id)
        if self._persist_path:
            self._save()
        return ref.ref_id

    def get(self, ref_id: str) -> Optional[EvidenceRef]:
        return self._refs.get(ref_id)

    def get_for_variable(self, variable: str) -> List[EvidenceRef]:
        """All evidence refs for a variable, newest first."""
        ids = self._by_variable.get(variable, [])
        refs = [self._refs[rid] for rid in ids if rid in self._refs]
        return sorted(refs, key=lambda r: r.retrieved_at, reverse=True)

    def get_for_source(self, source: str) -> List[EvidenceRef]:
        ids = self._by_source.get(source, [])
        return [self._refs[rid] for rid in ids if rid in self._refs]

    def best_evidence(self, variable: str) -> Optional[EvidenceRef]:
        """Highest-trust non-stale evidence for a variable."""
        refs = self.get_for_variable(variable)
        fresh = [r for r in refs if not r.is_stale()]
        if not fresh:
            fresh = refs  # fall back to stale if nothing else
        if not fresh:
            return None
        return max(fresh, key=lambda r: r.trust_score())

    def variables_with_evidence(self) -> List[str]:
        return list(self._by_variable.keys())

    def summary(self) -> Dict:
        total = len(self._refs)
        by_grade = {}
        for ref in self._refs.values():
            by_grade[ref.grade.value] = by_grade.get(ref.grade.value, 0) + 1
        stale = sum(1 for r in self._refs.values() if r.is_stale())
        return {
            "total_evidence_refs": total,
            "by_grade": by_grade,
            "stale_count": stale,
            "variables_covered": len(self._by_variable),
            "sources_used": len(self._by_source),
        }

    def to_audit_log(self) -> List[Dict]:
        return [ref.to_dict() for ref in
                sorted(self._refs.values(), key=lambda r: r.retrieved_at)]

    def _save(self):
        if self._persist_path:
            with open(self._persist_path, "w") as f:
                json.dump(self.to_audit_log(), f, indent=2, default=str)

    def _load(self):
        if self._persist_path and os.path.exists(self._persist_path):
            with open(self._persist_path) as f:
                data = json.load(f)
            for item in data:
                ref = EvidenceRef(
                    ref_id=item["ref_id"], source_name=item["source_name"],
                    method=EvidenceMethod(item["method"]),
                    grade=EvidenceGrade(item["grade"]),
                    retrieved_at=item["retrieved_at"],
                    payload_hash=item.get("payload_hash", ""),
                    variable=item["variable"], value=item["value"],
                    confidence=item.get("confidence", 0.5),
                )
                self._refs[ref.ref_id] = ref
                self._by_variable.setdefault(ref.variable, []).append(ref.ref_id)
                self._by_source.setdefault(ref.source_name, []).append(ref.ref_id)
