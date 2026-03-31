"""
pipeline.py — RPMPipeline: the top-level orchestrator.

Responsibility
--------------
RPMPipeline wires all agents and services together into a single, easy-to-use
object.  It is the main entry point for running the pipeline end-to-end.

Typical usage
-------------
    from veritas_rpm import RPMPipeline, PipelineConfig

    pipeline = RPMPipeline(config=PipelineConfig())

    # Feed raw data for a patient
    pipeline.ingest_ehr("patient-001", ehr_dict)
    pipeline.ingest_conversation("patient-001", conv_dict)
    pipeline.ingest_vitals("patient-001", vitals_dict)

    # Process the record through the full pipeline
    decision = pipeline.process("patient-001")

    # Retrieve queued nurse alerts
    nurse_queue = pipeline.dashboard.get_nurse_queue()

    # Inspect pipeline metrics
    print(pipeline.metrics.summary())

The pipeline instantiates and owns all components:
    VeritasAgent → SentinelLayer → DirectorAgent → MetaSentinelAgent → DashboardService

Dependencies are wired explicitly during construction so each component can
reference the collaborators it needs. This pattern also makes unit testing
straightforward: replace any component with a mock when constructing
RPMPipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from veritas_rpm.agents.director_agent import DirectorAgent
from veritas_rpm.agents.meta_sentinel_agent import MetaSentinelAgent
from veritas_rpm.agents.sentinel_layer import SentinelLayer
from veritas_rpm.agents.veritas_agent import VeritasAgent
from veritas_rpm.config import PipelineConfig
from veritas_rpm.metrics import PipelineMetrics
from veritas_rpm.models import (
    AgentClaim,
    CandidateAlert,
    SystemDecision,
    VeritasRecord,
)
from veritas_rpm.services.dashboard_service import DashboardService


class RPMPipeline:
    """
    End-to-end orchestrator for the Veritas-Sentinel-Director-MetaSentinel pipeline.

    Component wiring
    ----------------
    1. VeritasAgent — ingests raw data, emits VeritasRecord.
    2. SentinelLayer — subscribes to VeritasRecord stream, detects events,
       emits CandidateAlert to DirectorAgent.
    3. DirectorAgent — routes CandidateAlerts to specialist agents, collects
       AgentClaims, forwards to MetaSentinelAgent.
    4. MetaSentinelAgent — aggregates AgentClaims into SystemDecision, applies
       cooldown, routes to DashboardService.
    5. DashboardService — delivers SystemDecisions to the appropriate human role.

    Public access to components
    ---------------------------
    All five components are exposed as public attributes so callers can
    interact with them directly when needed:

        pipeline.veritas     — VeritasAgent
        pipeline.sentinel    — SentinelLayer
        pipeline.director    — DirectorAgent
        pipeline.meta        — MetaSentinelAgent
        pipeline.dashboard   — DashboardService
        pipeline.metrics     — PipelineMetrics
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        # Legacy parameter — prefer config
        cooldown_minutes: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Instantiate and wire all pipeline components.

        Parameters
        ----------
        config:
            Optional PipelineConfig controlling cooldown durations,
            routing table, and other tuneable parameters.
        cooldown_minutes:
            Legacy parameter.  Deprecated in favour of ``config``.
        """
        if config is None:
            config = PipelineConfig()

        self.metrics = PipelineMetrics()

        # Instantiate components bottom-up so dependencies can be injected.

        self.dashboard = DashboardService()

        # Legacy cooldown_minutes takes precedence over config.cooldown
        # when explicitly provided, for backward compatibility.
        self.meta = MetaSentinelAgent(
            cooldown_config=config.cooldown if cooldown_minutes is None else None,
            cooldown_minutes=cooldown_minutes,
            dashboard_service=self.dashboard,
            metrics=self.metrics,
        )

        self.director = DirectorAgent(
            meta_sentinel=self.meta,
            routing_config=config.routing,
            metrics=self.metrics,
        )

        self.sentinel = SentinelLayer(
            director=self.director,
            metrics=self.metrics,
        )

        self.veritas = VeritasAgent()

        # Wire VeritasAgent → SentinelLayer (pub/sub)
        # SentinelLayer.on_record() is the subscriber callback.
        # It also needs to update DirectorAgent's context cache, so we wrap it.
        def _sentinel_and_context_update(record: VeritasRecord) -> None:
            self.director.update_context(record)
            self.sentinel.on_record(record)

        self.veritas.subscribe(_sentinel_and_context_update)

        # Wire DashboardService feedback dependencies explicitly
        self.dashboard.set_veritas_agent(self.veritas)
        self.dashboard.set_meta_sentinel(self.meta)

    # ------------------------------------------------------------------
    # Convenience ingest delegates
    # ------------------------------------------------------------------

    def ingest_ehr(self, patient_id: str, raw: Dict[str, Any]) -> None:
        """Delegate to VeritasAgent.ingest_ehr()."""
        self.veritas.ingest_ehr(patient_id, raw)

    def ingest_conversation(self, patient_id: str, raw: Dict[str, Any]) -> None:
        """Delegate to VeritasAgent.ingest_conversation()."""
        self.veritas.ingest_conversation(patient_id, raw)

    def ingest_vitals(self, patient_id: str, raw: Dict[str, Any]) -> None:
        """Delegate to VeritasAgent.ingest_vitals()."""
        self.veritas.ingest_vitals(patient_id, raw)

    def ingest_patient_input(self, patient_id: str, raw: Dict[str, Any]) -> None:
        """Delegate to VeritasAgent.ingest_patient_input()."""
        self.veritas.ingest_patient_input(patient_id, raw)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, patient_id: str) -> VeritasRecord:
        """
        Build a VeritasRecord for the patient and run it through the pipeline.

        This triggers:
        1. VeritasAgent.build_record() → emits VeritasRecord
        2. Subscriber callback → SentinelLayer.on_record()
        3. SentinelLayer detects events → CandidateAlert(s) sent to DirectorAgent
        4. DirectorAgent routes to specialist agents → AgentClaim(s)
        5. DirectorAgent forwards to MetaSentinelAgent
        6. MetaSentinelAgent produces SystemDecision → DashboardService

        Parameters
        ----------
        patient_id:
            The patient for whom to run the pipeline.

        Returns
        -------
        VeritasRecord
            The assembled record (the rest of the pipeline runs via callbacks).

        Notes
        -----
        Because the pipeline uses a synchronous pub/sub model, all downstream
        processing is complete by the time this method returns.
        """
        return self.veritas.build_record(patient_id)

    def get_all_decisions(self) -> List[SystemDecision]:
        """
        Return all SystemDecisions currently pending across all role queues.

        Convenience method for inspecting pipeline output in examples and tests.
        """
        return (
            self.dashboard.get_patient_queue()
            + self.dashboard.get_nurse_queue()
            + self.dashboard.get_doctor_queue()
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all pipeline metrics."""
        return self.metrics.summary()
