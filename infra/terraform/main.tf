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

# The scorer would run on ECS/Fargate or Lambda (maps to the local FastAPI container).
# resource "aws_ecs_service" "scorer" { ... }
