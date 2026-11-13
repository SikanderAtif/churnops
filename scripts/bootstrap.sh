#!/usr/bin/env bash
# One-time setup: initialise DVC, fetch + track raw data, configure the S3 remote.
# Usage: bash scripts/bootstrap.sh <s3-bucket-name> [prefix]
set -euo pipefail

BUCKET="${1:?usage: bash scripts/bootstrap.sh <s3-bucket-name> [prefix]}"
PREFIX="${2:-dvcstore}"

[ -d .git ] || git init
[ -d .dvc ] || dvc init

python -m src.data.ingest
dvc add data/raw/telco_churn.csv

dvc remote add -d -f s3remote "s3://${BUCKET}/${PREFIX}"
echo "DVC remote set to s3://${BUCKET}/${PREFIX}"
echo "Next: dvc repro && dvc push && git add . && git commit -m 'Phase 1 pipeline'"
