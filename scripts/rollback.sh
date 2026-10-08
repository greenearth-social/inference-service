#!/bin/bash

# Green Earth Inference Service - Cloud Run rollback
#
# Shifts all traffic back to a previously deployed Cloud Run revision. Because
# deploy.sh uses source deploys (`gcloud run deploy --source=.`), the repo never
# names an image tag — each revision is the durable record of a deployment,
# pinning both the built image digest and the env config it ran with. Rolling
# back therefore means re-pointing traffic at an older revision, not rebuilding
# an older git sha (see greenearth-social/api#181).
#
# Model manifest URIs (GE_INFERENCE_*_MANIFEST_URI) are part of that per-revision
# env config, so a rollback restores the code and the model artifacts it was
# deployed against, together. That is what you want: rolling code back without
# its manifests would pair old code with newer model files.
#
# Rollbacks are manual. Cloud Run's own health-check behavior is untouched: a
# revision that never becomes Ready never receives traffic in the first place.
# Domain mappings (inference[-stage].greenearth.social) target the service, not a
# revision, so they follow the rollback with no extra work.
#
# After a rollback, traffic is pinned to a named revision. The next successful
# deploy.sh run resets traffic to LATEST, so "deploy the fix" is also how you
# leave the rolled-back state.

set -e

# Configuration (env vars, overridden by CLI args — same convention as deploy.sh)
GE_GCP_PROJECT_ID="${GE_GCP_PROJECT_ID:-greenearth-471522}"
GE_GCP_REGION="${GE_GCP_REGION:-us-east1}"
GE_ENVIRONMENT="${GE_ENVIRONMENT:-stage}"

# Rollback target: a revision name or a git sha. Empty means "the previous
# deployment", resolved automatically.
TARGET=""

LIST_ONLY=false
DRY_RUN=false
ASSUME_YES=false

# How long to wait for /health to report the rolled-back git sha.
HEALTH_TIMEOUT_SEC=90

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_action() {
    echo -e "${BLUE}[ROLLBACK]${NC} $1"
}

service_name() {
    echo "engagement-prediction-inference-$GE_ENVIRONMENT"
}

require_service() {
    if ! gcloud run services describe "$(service_name)" \
        --region="$GE_GCP_REGION" --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        log_error "Cloud Run service $(service_name) not found in $GE_GCP_REGION."
        log_error "Check --environment (stage/prod), --region, and --project-id."
        exit 1
    fi
}

serving_revision() {
    gcloud run services describe "$(service_name)" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --format="value(status.traffic.revisionName)" | head -n 1
}

revision_git_sha() {
    local revision="$1"
    gcloud run revisions describe "$revision" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --format="value(metadata.labels.git-sha)" 2>/dev/null || true
}

revision_env_var() {
    local revision="$1"
    local var_name="$2"
    gcloud run revisions describe "$revision" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --flatten="spec.containers[].env[]" \
        --format="value(spec.containers.env.name,spec.containers.env.value)" 2>/dev/null \
        | awk -F'\t' -v name="$var_name" '$1 == name { print $2; exit }'
}

# Ready revisions, newest first, as "name|git-sha|created". Pipe-separated
# rather than tab-separated because bash collapses runs of whitespace delimiters:
# an unlabelled revision would otherwise shift its timestamp into the sha field.
ready_revisions() {
    gcloud run revisions list \
        --service="$(service_name)" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --filter="status.conditions.type=Ready AND status.conditions.status=True" \
        --sort-by="~metadata.creationTimestamp" \
        --format="value[separator='|'](metadata.name,metadata.labels.git-sha,metadata.creationTimestamp)"
}

list_revisions() {
    local serving
    serving="$(serving_revision)"

    log_info "Rollback candidates for $(service_name) (newest first):"
    echo ""
    printf "    %-48s %-10s %s\n" "REVISION" "GIT SHA" "CREATED"

    while IFS='|' read -r revision sha created; do
        [ -z "$revision" ] && continue
        local marker="  "
        if [ "$revision" = "$serving" ]; then
            marker="=>"
        fi
        printf "%s  %-48s %-10s %s\n" "$marker" "$revision" "${sha:-(unstamped)}" "$created"
    done <<< "$(ready_revisions)"

    echo ""
    echo "  => currently serving traffic"
    echo "  (unstamped) = deployed before git-sha labelling; still a valid --to target by name"
    echo ""
    echo "  Roll back to the previous deployment:  $0 --environment $GE_ENVIRONMENT"
    echo "  Roll back to a specific target:        $0 --environment $GE_ENVIRONMENT --to <revision|git-sha>"
}

# Newest Ready revision older than the serving one whose git sha differs from
# what is serving now.
resolve_previous_revision() {
    local serving="$1"
    local serving_sha="$2"
    local seen_serving=false

    while IFS='|' read -r revision sha _created; do
        [ -z "$revision" ] && continue

        if [ "$revision" = "$serving" ]; then
            seen_serving=true
            continue
        fi

        # The list is newest-first, so anything before the serving entry is newer.
        [ "$seen_serving" = false ] && continue

        if [ -z "$sha" ] || [ -z "$serving_sha" ] || [ "$sha" != "$serving_sha" ]; then
            echo "$revision"
            return 0
        fi
    done <<< "$(ready_revisions)"

    return 1
}

# Accepts a revision name or a git sha; echoes the resolved revision name.
resolve_target_revision() {
    local requested="$1"

    if gcloud run revisions describe "$requested" \
        --region="$GE_GCP_REGION" --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        echo "$requested"
        return 0
    fi

    local match
    match="$(ready_revisions | awk -F'[|]' -v sha="$requested" '$2 == sha { print $1; exit }')"

    if [ -n "$match" ]; then
        echo "$match"
        return 0
    fi

    return 1
}

report_model_config() {
    local revision="$1"
    local label="$2"

    local models two_tower ranker
    models="$(revision_env_var "$revision" GE_INFERENCE_MODELS)"
    two_tower="$(revision_env_var "$revision" GE_INFERENCE_TWO_TOWER_MANIFEST_URI)"
    ranker="$(revision_env_var "$revision" GE_INFERENCE_RANKER_MANIFEST_URI)"

    echo "  $label models:   ${models:-(unset)}"
    echo "  $label two-tower: ${two_tower:-(unset)}"
    if [ -n "$ranker" ]; then
        echo "  $label ranker:    $ranker"
    fi
}

confirm_rollback() {
    local serving="$1"
    local serving_sha="$2"
    local target="$3"
    local target_sha="$4"

    echo ""
    if [ "$GE_ENVIRONMENT" = "prod" ]; then
        echo -e "${RED}*** PRODUCTION ROLLBACK ***${NC}"
    fi
    echo "  service:  $(service_name)  ($GE_ENVIRONMENT)"
    echo "  serving:  $serving  (${serving_sha:-unstamped})"
    echo "  target:   $target  (${target_sha:-unstamped})"
    echo ""
    echo "  Model configuration follows the revision:"
    report_model_config "$serving" "serving"
    report_model_config "$target" "target "
    echo ""
    log_warn "The API sends inference requests here. If the target revision serves"
    log_warn "different models, check that the deployed API still expects them."
    echo ""

    if [ "$ASSUME_YES" = true ]; then
        return 0
    fi

    local reply
    read -r -p "Shift 100% of traffic to the target revision? Type 'yes' to confirm: " reply
    if [ "$reply" != "yes" ]; then
        log_info "Aborted — nothing changed."
        exit 0
    fi
}

shift_traffic() {
    local target="$1"

    local cmd="gcloud run services update-traffic $(service_name)"
    cmd="$cmd --region=$GE_GCP_REGION"
    cmd="$cmd --project=$GE_GCP_PROJECT_ID"
    cmd="$cmd --to-revisions=$target=100"

    if [ "$DRY_RUN" = true ]; then
        log_info "[dry run] would execute:"
        echo "  $cmd"
        return 0
    fi

    log_action "Executing: $cmd"
    if ! eval "$cmd" > /dev/null; then
        log_error "Failed to shift traffic to $target"
        exit 1
    fi

    log_info "✓ Traffic now served by $target"
}

# Confirms the rollback took effect from outside Cloud Run's own bookkeeping, by
# asking the running service which git sha it is. /health is unauthenticated;
# model loading happens lazily, so a fresh instance may take a moment to answer.
verify_health() {
    local target_sha="$1"

    if [ -z "$target_sha" ]; then
        log_warn "Target revision has no git-sha label — skipping /health verification."
        log_warn "Revisions deployed before git-sha stamping cannot self-report."
        return 0
    fi

    local service_url
    service_url=$(gcloud run services describe "$(service_name)" \
        --region="$GE_GCP_REGION" --project="$GE_GCP_PROJECT_ID" --format="value(status.url)")

    log_info "Verifying $service_url/health reports git sha $target_sha..."

    local deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
    while [ "$SECONDS" -lt "$deadline" ]; do
        local reported
        reported=$(curl -fsS --max-time 10 "$service_url/health" 2>/dev/null \
            | sed -n 's/.*"git_sha"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')

        if [ "$reported" = "$target_sha" ]; then
            log_info "✓ /health reports git sha $reported"
            log_info "Model readiness is separate — check /ready (needs X-API-Key) if in doubt."
            return 0
        fi

        sleep 3
    done

    log_warn "/health did not report git sha $target_sha within ${HEALTH_TIMEOUT_SEC}s."
    log_warn "Traffic was shifted; check the service manually before assuming the rollback failed."
    return 0
}

main() {
    log_info "Green Earth inference service rollback"
    log_info "Project:     $GE_GCP_PROJECT_ID"
    log_info "Region:      $GE_GCP_REGION"
    log_info "Environment: $GE_ENVIRONMENT"

    require_service

    if [ "$LIST_ONLY" = true ]; then
        list_revisions
        exit 0
    fi

    local serving serving_sha target target_sha
    serving="$(serving_revision)"
    if [ -z "$serving" ]; then
        log_error "Could not determine which revision is serving traffic."
        exit 1
    fi
    serving_sha="$(revision_git_sha "$serving")"

    if [ -n "$TARGET" ]; then
        if ! target="$(resolve_target_revision "$TARGET")"; then
            log_error "No revision matches '$TARGET' (tried revision name, then git sha)."
            log_error "Run '$0 --environment $GE_ENVIRONMENT --list' to see candidates."
            exit 1
        fi
    else
        if ! target="$(resolve_previous_revision "$serving" "$serving_sha")"; then
            log_error "No previous Ready revision found to roll back to."
            log_error "Run '$0 --environment $GE_ENVIRONMENT --list' to see what exists."
            exit 1
        fi
    fi

    if [ "$target" = "$serving" ]; then
        log_info "$target is already serving traffic — nothing to do."
        exit 0
    fi

    target_sha="$(revision_git_sha "$target")"

    confirm_rollback "$serving" "$serving_sha" "$target" "$target_sha"
    shift_traffic "$target"

    if [ "$DRY_RUN" = true ]; then
        log_info "[dry run] no changes made."
        exit 0
    fi

    verify_health "$target_sha"

    echo ""
    log_info "To leave the rolled-back state, deploy the fix normally:"
    echo "  ./scripts/deploy.sh --environment $GE_ENVIRONMENT ..."
    echo "  (deploy.sh resets traffic to LATEST on success)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            GE_GCP_PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            GE_GCP_REGION="$2"
            shift 2
            ;;
        --environment)
            GE_ENVIRONMENT="$2"
            shift 2
            ;;
        --to)
            TARGET="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes)
            ASSUME_YES=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Rolls the inference service back to a previously deployed Cloud Run"
            echo "revision by shifting 100% of traffic to it. With no --to, the previous"
            echo "deployment (newest Ready revision older than the one serving, with a"
            echo "different git sha) is used. Model manifest URIs travel with the"
            echo "revision, so code and model artifacts roll back together."
            echo ""
            echo "Options:"
            echo "  --environment ENV        Environment name (default: stage)"
            echo "  --to REVISION|GIT_SHA    Roll back to a specific revision or git sha"
            echo "  --list                   List rollback candidates and exit"
            echo "  --dry-run                Show what would change, execute nothing"
            echo "  --yes                    Skip the confirmation prompt"
            echo "  --project-id ID          GCP project ID (default: greenearth-471522)"
            echo "  --region REGION          GCP region (default: us-east1)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GE_ENVIRONMENT           Same as --environment"
            echo "  GE_GCP_PROJECT_ID        Same as --project-id"
            echo "  GE_GCP_REGION            Same as --region"
            echo ""
            echo "Examples:"
            echo "  $0 --environment prod --list"
            echo "  $0 --environment prod"
            echo "  $0 --environment prod --to 7176a35"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

main
