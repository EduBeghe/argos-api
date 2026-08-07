from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.models import MLRefinementRunRecord
from app.ml.refinement.persistence import experiment_key
from app.ml.refinement.planning import RefinementPlan, build_refinement_plan

from .contracts import CAMPAIGN_CONFIDENCE_BUCKETS


def reset_refinement_run(
    run: MLRefinementRunRecord,
) -> tuple[MLRefinementRequest, RefinementPlan]:
    """Reset one persisted child while preserving reusable evaluation evidence."""
    payload = dict(run.normalized_configuration)
    if payload.get("economic_policy_search") is not None:
        payload["confidence_buckets"] = [
            str(item) for item in CAMPAIGN_CONFIDENCE_BUCKETS
        ]
    payload["notes"] = run.notes
    request = MLRefinementRequest.model_validate(payload)
    plan = build_refinement_plan(request)
    if not plan.evaluations:
        raise ValueError("persisted refinement plan selected no evaluations")
    run.status = "queued"
    run.experiment_key = experiment_key(
        request=request,
        dataset_fingerprint=run.dataset_fingerprint,
        versions=run.library_versions,
    )
    run.normalized_configuration = request.normalized_payload()
    run.maximum_trials = request.maximum_trials
    run.maximum_runtime_seconds = request.maximum_runtime_seconds
    run.combinations_planned = plan.total_combinations
    run.evaluations_selected = len(plan.evaluations)
    run.evaluations_computed = 0
    run.evaluations_reused = 0
    run.evaluations_omitted = plan.omitted_by_trial_budget
    run.budget_exhausted = plan.omitted_by_trial_budget > 0
    run.termination_reason = None
    run.started_at = None
    run.completed_at = None
    run.failed_at = None
    run.failure_type = None
    run.failure_message = None
    run.elapsed_seconds = None
    run.worker_id = None
    run.claimed_at = None
    run.heartbeat_at = None
    run.cancellation_requested_at = None
    return request, plan
