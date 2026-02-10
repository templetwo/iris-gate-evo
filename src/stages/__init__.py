"""S1-S3 stage orchestration — Formulation, Synthesis, Stable Attractor."""

from src.stages.stages import (
    run_pipeline,
    run_pipeline_sync,
    run_s3_gate,
    build_recirculation_context,
    enrich_compiled_for_recirculation,
)
from src.stages.synthesis import run_s2_synthesis, SynthesizedClaim
