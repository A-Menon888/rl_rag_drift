---
updated: 2025-03-10
---
# Deployment Reference

## Packaging

The application is deployed as a container image based on `debian-slim`.

Deployment artifacts are stored in the `registry` repository.

## Orchestration

Deployments are orchestrated with `Kubernetes`.

The primary deployment region is `us-east-1`.

The default number of replicas is `3`.

Releases use a `rolling` deployment strategy.

Deployments are frozen on `Fridays`.

## Resources

Each instance is limited to `2` CPU cores.

Each instance is limited to `4 GB` of memory.

## Health Checks

The health check endpoint is `/healthz`.

The readiness probe runs every `10` seconds.

## Secrets and Backups

Secrets are loaded from `env-files`.

Database backups run every `24` hours.
