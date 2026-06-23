#!/bin/bash

# Green Earth Engagement Prediction - Inference Service Cloud Run Deployment Script
# Deploys the inference FastAPI service to Google Cloud Run using source deployment.
# The CPU Dockerfile in this directory is picked up automatically.
#
# Prerequisites: Run gcp_setup.sh first to configure the GCP environment.
# Must be run from within the inference_service/ directory.

set -e

# Configuration
GE_GCP_PROJECT_ID="${GE_GCP_PROJECT_ID:-greenearth-471522}"
GE_GCP_REGION="${GE_GCP_REGION:-us-east1}"
GE_ENVIRONMENT="${GE_ENVIRONMENT:-stage}"
GE_ENABLE_INFERENCE_DOMAIN_MAPPING="${GE_ENABLE_INFERENCE_DOMAIN_MAPPING:-true}"
GE_INFERENCE_DOMAIN="${GE_INFERENCE_DOMAIN:-}"
GE_INFERENCE_MIN_INSTANCES="${GE_INFERENCE_MIN_INSTANCES:-1}"
GE_INFERENCE_MAX_INSTANCES="${GE_INFERENCE_MAX_INSTANCES:-1}"

# Multi-model config — required, no defaults
GE_INFERENCE_MODELS="${GE_INFERENCE_MODELS:-}"
GE_INFERENCE_TWO_TOWER_MANIFEST_URI="${GE_INFERENCE_TWO_TOWER_MANIFEST_URI:-}"
GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI="${GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI:-}"
GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN="${GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN:-128}"
GE_INFERENCE_RANKER_MANIFEST_URI="${GE_INFERENCE_RANKER_MANIFEST_URI:-}"
GE_INFERENCE_RANKER_AUTHOR_MAP_URI="${GE_INFERENCE_RANKER_AUTHOR_MAP_URI:-}"
GE_INFERENCE_RANKER_MAX_HISTORY_LEN="${GE_INFERENCE_RANKER_MAX_HISTORY_LEN:-}"
GE_INFERENCE_CONTENT_EMBED_DIM="${GE_INFERENCE_CONTENT_EMBED_DIM:-384}"
GE_INFERENCE_MAX_BATCH="${GE_INFERENCE_MAX_BATCH:-0}"

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

log_build() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

models_include() {
    local model_type="$1"
    local models="${GE_INFERENCE_MODELS// /}"
    case ",$models," in
        *",$model_type,"*) return 0 ;;
        *) return 1 ;;
    esac
}

get_domain_mapping_condition_status() {
    local domain="$1"
    local condition_type="$2"

    gcloud beta run domain-mappings describe --domain="$domain" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --format=json 2>/dev/null | python3 -c '
import json
import sys

condition_type = sys.argv[1]

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)

for condition in payload.get("status", {}).get("conditions", []):
    if condition.get("type") == condition_type:
        print(condition.get("status", ""))
        break
else:
    print("")
' "$condition_type"
}

default_inference_domain() {
    if [ "$GE_ENVIRONMENT" = "prod" ]; then
        echo "inference.greenearth.social"
    else
        echo "inference-stage.greenearth.social"
    fi
}

resolve_inference_domain() {
    if [ -n "$GE_INFERENCE_DOMAIN" ]; then
        echo "$GE_INFERENCE_DOMAIN"
        return
    fi
    echo "$(default_inference_domain)"
}

generate_requirements() {
    log_info "Generating requirements.txt from Pipfile..."

    if ! command -v pipenv &> /dev/null; then
        log_error "pipenv is not installed. Please install it first: pip install pipenv"
        exit 1
    fi

    pipenv requirements > requirements.txt

    if [ $? -eq 0 ]; then
        log_info "Generated requirements.txt successfully"
    else
        log_error "Failed to generate requirements.txt"
        exit 1
    fi
}

validate_config() {
    log_info "Validating configuration..."

    if [ -z "$GE_INFERENCE_MODELS" ]; then
        log_error "GE_INFERENCE_MODELS is required. Set it via env var or --models flag."
        log_error "Example: GE_INFERENCE_MODELS=user-tower,post-tower ./deploy.sh"
        exit 1
    fi

    if [ -z "$GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN" ] || ! [[ "$GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN" =~ ^[1-9][0-9]*$ ]]; then
        log_error "GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN is required and must be a positive integer."
        log_error "Example: GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN=128 ./deploy.sh"
        exit 1
    fi

    if [ -z "$GE_INFERENCE_CONTENT_EMBED_DIM" ] || ! [[ "$GE_INFERENCE_CONTENT_EMBED_DIM" =~ ^[1-9][0-9]*$ ]]; then
        log_error "GE_INFERENCE_CONTENT_EMBED_DIM is required and must be a positive integer."
        log_error "Example: GE_INFERENCE_CONTENT_EMBED_DIM=384 ./deploy.sh"
        exit 1
    fi

    if [ -z "$GE_INFERENCE_TWO_TOWER_MANIFEST_URI" ]; then
        log_error "GE_INFERENCE_TWO_TOWER_MANIFEST_URI is required."
        log_error "Example: GE_INFERENCE_TWO_TOWER_MANIFEST_URI=gs://my-bucket/.../two_tower_serving_manifest.json ./deploy.sh"
        exit 1
    fi

    if [ -z "$GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI" ]; then
        log_error "GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI is required."
        log_error "Example: GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI=gs://my-bucket/author_idx.parquet ./deploy.sh"
        exit 1
    fi

    if models_include "ranker"; then
        if [ -z "$GE_INFERENCE_RANKER_MANIFEST_URI" ]; then
            log_error "GE_INFERENCE_RANKER_MANIFEST_URI is required when GE_INFERENCE_MODELS includes ranker."
            log_error "Example: GE_INFERENCE_RANKER_MANIFEST_URI=gs://my-bucket/.../ranker_serving_manifest.json ./deploy.sh"
            exit 1
        fi

        if [ -z "$GE_INFERENCE_RANKER_AUTHOR_MAP_URI" ]; then
            log_error "GE_INFERENCE_RANKER_AUTHOR_MAP_URI is required when GE_INFERENCE_MODELS includes ranker."
            log_error "Example: GE_INFERENCE_RANKER_AUTHOR_MAP_URI=gs://my-bucket/ranker_author_idx.parquet ./deploy.sh"
            exit 1
        fi

        if [ -z "$GE_INFERENCE_RANKER_MAX_HISTORY_LEN" ] || ! [[ "$GE_INFERENCE_RANKER_MAX_HISTORY_LEN" =~ ^[1-9][0-9]*$ ]]; then
            log_error "GE_INFERENCE_RANKER_MAX_HISTORY_LEN is required and must be a positive integer when GE_INFERENCE_MODELS includes ranker."
            log_error "Example: GE_INFERENCE_RANKER_MAX_HISTORY_LEN=128 ./deploy.sh"
            exit 1
        fi
    fi

    if [ -z "$GE_INFERENCE_MAX_BATCH" ] || ! [[ "$GE_INFERENCE_MAX_BATCH" =~ ^[0-9]+$ ]]; then
        log_error "GE_INFERENCE_MAX_BATCH must be a non-negative integer."
        log_error "Example: GE_INFERENCE_MAX_BATCH=0 ./deploy.sh"
        exit 1
    fi

    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" = "true" ]; then
        log_info "Inference domain mapping enabled for: $(resolve_inference_domain)"
    else
        log_info "Inference domain mapping disabled"
    fi

    gcloud config set project "$GE_GCP_PROJECT_ID"

    log_info "Configuration validation complete."
}

reconcile_domain_mapping() {
    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" != "true" ]; then
        return
    fi

    local service_name="engagement-prediction-inference-$GE_ENVIRONMENT"
    local domain
    domain="$(resolve_inference_domain)"
    local mapped_route_name

    log_info "Reconciling domain mapping for $domain"

    if gcloud beta run domain-mappings describe --domain="$domain" --region="$GE_GCP_REGION" --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        mapped_route_name=$(gcloud beta run domain-mappings describe \
            --domain="$domain" \
            --region="$GE_GCP_REGION" \
            --project="$GE_GCP_PROJECT_ID" \
            --format="value(status.mappedRouteName)" 2>/dev/null || echo "")

        if [ -z "$mapped_route_name" ]; then
            mapped_route_name=$(gcloud beta run domain-mappings describe \
                --domain="$domain" \
                --region="$GE_GCP_REGION" \
                --project="$GE_GCP_PROJECT_ID" \
                --format="value(spec.routeName)" 2>/dev/null || echo "")
        fi

        if [ "$mapped_route_name" = "$service_name" ]; then
            log_info "Domain mapping already targets expected service: $service_name"
        else
            log_warn "Domain mapping exists but points to '$mapped_route_name'; overriding to '$service_name'"
            gcloud beta run domain-mappings create \
                --service="$service_name" \
                --domain="$domain" \
                --force-override \
                --region="$GE_GCP_REGION" \
                --project="$GE_GCP_PROJECT_ID"
            log_info "Updated domain mapping: $domain -> $service_name"
        fi
    else
        gcloud beta run domain-mappings create \
            --service="$service_name" \
            --domain="$domain" \
            --region="$GE_GCP_REGION" \
            --project="$GE_GCP_PROJECT_ID"
        log_info "Created domain mapping: $domain"
    fi

    local mapping_ready
    mapping_ready=$(get_domain_mapping_condition_status "$domain" "Ready")
    if [ "$mapping_ready" = "True" ]; then
        log_info "Mapped URL: https://$domain (Ready)"
    else
        log_warn "Mapped URL: https://$domain (not Ready yet; check DNS/certificate provisioning)"
    fi
}

verify_vpc_connector() {
    log_info "Verifying VPC connector exists..."

    local connector_name="ingex-vpc-connector-$GE_ENVIRONMENT"

    if ! gcloud compute networks vpc-access connectors describe "$connector_name" --region="$GE_GCP_REGION" > /dev/null 2>&1; then
        log_warn "VPC connector '$connector_name' does not exist"
        log_warn "Deploying without VPC connector — service will not have internal network access"
        log_warn "Run ../ingex/ingest/scripts/gcp_setup.sh to create the VPC connector if needed"
        VPC_CONNECTOR_EXISTS=false
    else
        local connector_status
        connector_status=$(gcloud compute networks vpc-access connectors describe "$connector_name" --region="$GE_GCP_REGION" --format="value(state)" 2>/dev/null || echo "UNKNOWN")

        if [ "$connector_status" != "READY" ]; then
            log_warn "VPC connector '$connector_name' is not ready (status: $connector_status)"
            log_warn "This may cause deployment to fail. Wait a few minutes and try again."
        else
            log_info "VPC connector '$connector_name' is ready"
        fi
        VPC_CONNECTOR_EXISTS=true
    fi
}

deploy_inference_service() {
    local service_name="engagement-prediction-inference-$GE_ENVIRONMENT"
    local sa_email="engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"

    log_info "Deploying $service_name from source..."

    # Write env vars to a temp YAML file so values with spaces or special
    # characters (e.g. GCS paths) are passed safely to gcloud.
    local temp_var_dir
    temp_var_dir=$(mktemp -d)
    trap "rm -rf $temp_var_dir" EXIT
    cat > "$temp_var_dir/env-vars.yaml" <<EOF
GE_INFERENCE_MODELS: "$GE_INFERENCE_MODELS"
GE_INFERENCE_TWO_TOWER_MANIFEST_URI: "$GE_INFERENCE_TWO_TOWER_MANIFEST_URI"
GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN: "$GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN"
GE_INFERENCE_RANKER_MANIFEST_URI: "$GE_INFERENCE_RANKER_MANIFEST_URI"
GE_INFERENCE_RANKER_AUTHOR_MAP_URI: "$GE_INFERENCE_RANKER_AUTHOR_MAP_URI"
GE_INFERENCE_RANKER_MAX_HISTORY_LEN: "$GE_INFERENCE_RANKER_MAX_HISTORY_LEN"
GE_INFERENCE_CONTENT_EMBED_DIM: "$GE_INFERENCE_CONTENT_EMBED_DIM"
GE_INFERENCE_MAX_BATCH: "$GE_INFERENCE_MAX_BATCH"
GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI: "$GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI"
GE_INFERENCE_PREFER_CUDA: "0"
GE_INFERENCE_WARMUP: "0"
EOF

    local deploy_cmd="gcloud run deploy $service_name"
    deploy_cmd="$deploy_cmd --source=."
    deploy_cmd="$deploy_cmd --region=$GE_GCP_REGION"
    deploy_cmd="$deploy_cmd --service-account=$sa_email"

    if [ "$VPC_CONNECTOR_EXISTS" = true ]; then
        deploy_cmd="$deploy_cmd --vpc-connector=ingex-vpc-connector-$GE_ENVIRONMENT"
        deploy_cmd="$deploy_cmd --vpc-egress=private-ranges-only"
    fi

    deploy_cmd="$deploy_cmd --ingress=all"
    deploy_cmd="$deploy_cmd --allow-unauthenticated"
    deploy_cmd="$deploy_cmd --set-secrets=GE_INFERENCE_API_KEY=inference-api-key-$GE_ENVIRONMENT:latest"
    deploy_cmd="$deploy_cmd --env-vars-file=$temp_var_dir/env-vars.yaml"

    deploy_cmd="$deploy_cmd --cpu=2"
    deploy_cmd="$deploy_cmd --memory=2Gi"
    deploy_cmd="$deploy_cmd --timeout=120"
    deploy_cmd="$deploy_cmd --min-instances=$GE_INFERENCE_MIN_INSTANCES"
    deploy_cmd="$deploy_cmd --max-instances=$GE_INFERENCE_MAX_INSTANCES"

    log_build "Executing: $deploy_cmd"
    eval "$deploy_cmd"

    log_info "✓ $service_name deployed successfully"

    local service_url
    service_url=$(gcloud run services describe "$service_name" --region="$GE_GCP_REGION" --format="value(status.url)")
    log_info "Service URL: $service_url"
}

main() {
    log_info "Starting engagement prediction inference service deployment..."
    log_info "Project:         $GE_GCP_PROJECT_ID"
    log_info "Region:          $GE_GCP_REGION"
    log_info "Environment:     $GE_ENVIRONMENT"
    log_info "Models:          $GE_INFERENCE_MODELS"
    log_info "Manifest URI:    $GE_INFERENCE_TWO_TOWER_MANIFEST_URI"
    log_info "Max history len: $GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN"
    if models_include "ranker"; then
        log_info "Ranker manifest: $GE_INFERENCE_RANKER_MANIFEST_URI"
        log_info "Ranker max hist: $GE_INFERENCE_RANKER_MAX_HISTORY_LEN"
    fi
    log_info "Embed dimension: $GE_INFERENCE_CONTENT_EMBED_DIM"
    log_info "Max batch:       $GE_INFERENCE_MAX_BATCH"

    validate_config
    generate_requirements
    verify_vpc_connector
    deploy_inference_service
    reconcile_domain_mapping

    log_info "Deployment complete!"
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
        --models)
            GE_INFERENCE_MODELS="$2"
            shift 2
            ;;
        --two-tower-manifest-uri)
            GE_INFERENCE_TWO_TOWER_MANIFEST_URI="$2"
            shift 2
            ;;
        --two-tower-author-map-uri)
            GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI="$2"
            shift 2
            ;;
        --two-tower-max-history-len)
            GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN="$2"
            shift 2
            ;;
        --ranker-manifest-uri)
            GE_INFERENCE_RANKER_MANIFEST_URI="$2"
            shift 2
            ;;
        --ranker-author-map-uri)
            GE_INFERENCE_RANKER_AUTHOR_MAP_URI="$2"
            shift 2
            ;;
        --ranker-max-history-len)
            GE_INFERENCE_RANKER_MAX_HISTORY_LEN="$2"
            shift 2
            ;;
        --content-embed-dim)
            GE_INFERENCE_CONTENT_EMBED_DIM="$2"
            shift 2
            ;;
        --max-batch)
            GE_INFERENCE_MAX_BATCH="$2"
            shift 2
            ;;
        --inference-domain)
            GE_INFERENCE_DOMAIN="$2"
            shift 2
            ;;
        --disable-domain-mapping)
            GE_ENABLE_INFERENCE_DOMAIN_MAPPING="false"
            shift
            ;;
        --min-instances)
            GE_INFERENCE_MIN_INSTANCES="$2"
            shift 2
            ;;
        --max-instances)
            GE_INFERENCE_MAX_INSTANCES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project-id ID                 GCP project ID (default: greenearth-471522)"
            echo "  --region REGION                 GCP region (default: us-east1)"
            echo "  --environment ENV               Environment name (default: stage)"
            echo "  --models TYPES                  Comma-separated model types to load (required)"
            echo "                                  Supported: user-tower, post-tower, ranker"
            echo "  --two-tower-manifest-uri URI    GCS URI or local path to two_tower_serving_manifest.json (required)"
            echo "  --content-embed-dim N           Dimension of the input content embeddings (required)"
            echo "  --two-tower-author-map-uri URI  GCS URI or local path for the author idx parquet map (required)"
            echo "  --two-tower-max-history-len N   Maximum user history sequence length (required)"
            echo "  --ranker-manifest-uri URI       GCS URI or local path to ranker_serving_manifest.json (required with ranker)"
            echo "  --ranker-author-map-uri URI     GCS URI or local path for the ranker author idx parquet map (required with ranker)"
            echo "  --ranker-max-history-len N      Maximum ranker history sequence length (required with ranker)"
            echo "  --max-batch N                   Max batch size allowed for inference"
            echo "  --inference-domain DOMAIN       Custom mapped domain for inference service"
            echo "  --disable-domain-mapping        Skip domain mapping reconciliation"
            echo "  --min-instances N               Minimum Cloud Run instances (default: 1)"
            echo "  --max-instances N               Maximum Cloud Run instances (default: 1)"
            echo "  --help                          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GE_GCP_PROJECT_ID                        Same as --project-id"
            echo "  GE_GCP_REGION                            Same as --region"
            echo "  GE_ENVIRONMENT                           Same as --environment"
            echo "  GE_INFERENCE_MODELS                      Same as --models (required)"
            echo "  GE_INFERENCE_TWO_TOWER_MANIFEST_URI      Same as --two-tower-manifest-uri (required)"
            echo "  GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN   Same as --two-tower-max-history-len (required)"
            echo "  GE_INFERENCE_RANKER_MANIFEST_URI         Same as --ranker-manifest-uri (required with ranker)"
            echo "  GE_INFERENCE_RANKER_AUTHOR_MAP_URI       Same as --ranker-author-map-uri (required with ranker)"
            echo "  GE_INFERENCE_RANKER_MAX_HISTORY_LEN      Same as --ranker-max-history-len (required with ranker)"
            echo "  GE_INFERENCE_CONTENT_EMBED_DIM           Same as --content-embed-dim (required)"
            echo "  GE_INFERENCE_MAX_BATCH                   Same as --max-batch (optional; 0 = no limit)"
            echo "  GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI    GCS URI or local path for the two tower author idx parquet map"
            echo "  GE_ENABLE_INFERENCE_DOMAIN_MAPPING       true/false toggle (default: true)"
            echo "  GE_INFERENCE_DOMAIN                      Custom mapped domain"
            echo "  GE_INFERENCE_MIN_INSTANCES               Minimum Cloud Run instances (default: 1)"
            echo "  GE_INFERENCE_MAX_INSTANCES               Maximum Cloud Run instances (default: 1)"
            echo ""
            echo "Examples:"
            echo "  $0 --environment stage \\"
            echo "     --models user-tower,post-tower \\"
            echo "     --two-tower-manifest-uri gs://my-bucket/.../two_tower_serving_manifest.json \\"
            echo "     --two-tower-author-map-uri gs://my-bucket/author_idx.parquet \\"
            echo "     --two-tower-max-history-len 50"
            echo ""
            echo "  $0 --environment stage \\"
            echo "     --models user-tower,post-tower,ranker \\"
            echo "     --two-tower-manifest-uri gs://my-bucket/.../two_tower_serving_manifest.json \\"
            echo "     --two-tower-author-map-uri gs://my-bucket/two_tower_author_idx.parquet \\"
            echo "     --two-tower-max-history-len 128 \\"
            echo "     --ranker-manifest-uri gs://my-bucket/.../ranker_serving_manifest.json \\"
            echo "     --ranker-author-map-uri gs://my-bucket/ranker_author_idx.parquet \\"
            echo "     --ranker-max-history-len 128"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

main
