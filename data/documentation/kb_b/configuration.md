---
updated: 2026-05-25
---
# Configuration Reference

## Environment

The application runs in the `production` environment by default.

The environment is selected with the `APP_ENV` variable.

Valid environment values are `development`, `staging`, `production`, and `preview`.

## Server Configuration

The application listens on port `8081`.

The server request timeout is `60` seconds.

The maximum request body size is `25 MB`.

## Database Configuration

The database connection pool has a maximum size of `40` connections.

The database connection timeout is `10` seconds.

## Logging

The default log level is `WARN`.

The application writes logs in JSON format.

## Feature Flags

The new dashboard is enabled by default.

The feature flag for the new dashboard is `ENABLE_NEW_DASHBOARD`.

The application also supports the `ENABLE_BULK_IMPORT` feature flag.