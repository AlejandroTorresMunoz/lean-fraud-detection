# PRODUCTION IaC — and it ACTUALLY APPLIES to the local stack via `tflocal`.
# LocalStack emulates the AWS control plane (CreateStream, CreateBucket, IAM, ...), so this
# Terraform really provisions the resources the demo uses:
#   - Local (free):  tflocal init && tflocal apply -auto-approve   # -> resources live in LocalStack
#   - Real AWS:      terraform init && terraform apply             # -> needs a billed AWS account
# `tflocal` (pip install terraform-local) injects the http://localhost:4566 endpoints, so the
# resource definitions below stay vanilla/production-grade — no LocalStack-specific hacks.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" {
  type    = string
  default = "eu-west-1"
}

variable "artifacts_bucket" {
  type    = string
  default = "lean-fraud-artifacts"
}

provider "aws" {
  region = var.region
}

# Real-time transaction stream + fraud alerts.
resource "aws_kinesis_stream" "transactions" {
  name        = "tx-stream"
  shard_count = 1
}

resource "aws_kinesis_stream" "alerts" {
  name        = "alerts-stream"
  shard_count = 1
}

# Model artifacts + datasets.
resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket
}

# --- Monitoring: fraud-rate-spike alarm -------------------------------------------------------
# The stream consumer emits a `FraudAlertRate` custom metric (share of scored tx that flagged) to the
# LeanFraud namespace. This alarm fires when that rate spikes over a window — the one thing actionable
# live: with no ground-truth labels at scoring time we can't monitor F1/PR-AUC in real time, so a
# sudden jump in the alert rate is the signal for an attack or data drift. (Latency is a settled
# non-issue at ~10x headroom, so there is deliberately no latency alarm.)
#
# It provisions for real via `tflocal` against LocalStack, exactly like the streams/bucket. Note:
# LocalStack Community does not fully evaluate alarm state automatically — to see it fire in the demo
# you may push a data point (`awslocal cloudwatch set-alarm-state ...`). The *definition* is the
# production-grade artifact; the same HCL would alarm for real on AWS.
resource "aws_sns_topic" "fraud_alerts" {
  name = "fraud-rate-alarms"
}

resource "aws_cloudwatch_metric_alarm" "fraud_rate_spike" {
  alarm_name          = "fraud-rate-spike"
  alarm_description   = "Fraud-alert rate is abnormally high over the last few minutes (attack/drift)."
  namespace           = "LeanFraud"
  metric_name         = "FraudAlertRate"
  statistic           = "Average"
  period              = 60
  evaluation_periods  = 3
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0.05 # ~10x the ~0.5% base fraud rate
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.fraud_alerts.arn]
}

# The scorer would run on ECS/Fargate or Lambda (maps to the local FastAPI container).
# resource "aws_ecs_service" "scorer" { ... }
