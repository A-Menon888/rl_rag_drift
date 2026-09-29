---
updated: 2026-05-28
---
# Deployment Reference

## Packaging

The application is deployed as a container image based on `alpine`.

Deployment artifacts are stored in the `registry` repository.

Deployment artifacts are signed with `cosign`.

## Orchestration

Deployments are orchestrated with `Kubernetes`.

The primary deployment region is `us-east-1`.

The default number of replicas is `5`.

Releases use a `canary` deployment strategy.

Horizontal autoscaling is triggered above `70` percent CPU utilization.

## Resources

Each instance is limited to `4` CPU cores.

Each instance is limited to `8 GB` of memory.

## Health Checks

The health check endpoint is `/healthz`.

The readiness probe runs every `5` seconds.

## Secrets and Backups

Secrets are loaded from `vault`.

Database backups run every `6` hours.
