#!/usr/bin/env bash
# Provision the LocalStack resources used by the demo. LocalStack emulates the AWS control
# plane, so we apply the REAL Terraform via `tflocal`
# (pip install terraform-local). Falls back to awslocal if tflocal isn't installed.
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
BUCKET="${S3_BUCKET:-lean-fraud-artifacts}"
TX_STREAM="${KINESIS_TX_STREAM:-tx-stream}"
ALERTS_STREAM="${KINESIS_ALERTS_STREAM:-alerts-stream}"

echo "Waiting for LocalStack at ${ENDPOINT} ..."
until curl -sf "${ENDPOINT}/_localstack/health" >/dev/null 2>&1; do
  sleep 1
done

if command -v tflocal >/dev/null 2>&1; then
  echo "Applying Terraform with tflocal (real control-plane provisioning) ..."
  tflocal -chdir=infra/terraform init -input=false
  tflocal -chdir=infra/terraform apply -auto-approve
elif command -v awslocal >/dev/null 2>&1; then
  echo "tflocal not found — falling back to awslocal ..."
  awslocal kinesis create-stream --stream-name "${TX_STREAM}" --shard-count 1 || true
  awslocal kinesis create-stream --stream-name "${ALERTS_STREAM}" --shard-count 1 || true
  awslocal s3 mb "s3://${BUCKET}" || true
else
  echo "ERROR: neither tflocal nor awslocal is installed."
  echo "  pip install terraform-local awscli-local"
  exit 1
fi

echo "LocalStack ready. Kinesis: ${TX_STREAM}, ${ALERTS_STREAM}  |  S3 bucket: ${BUCKET}"
