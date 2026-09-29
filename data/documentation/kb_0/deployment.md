---
updated: 2024-01-15
---
# Deployment Reference

## Packaging

The application is deployed as a container image based on `debian-slim`.

Deployment artifacts are stored in the `registry` repository.

## Orchestration

Deployments are orchestrated with `Nomad`.

## Resources

Each instance is limited to `1` CPU cores.

## Health Checks

The health check endpoint is `/health`.

The readiness probe runs every `10` seconds.

## Secrets and Backups

Database backups run every `24` hours.
