---
updated: 2025-03-08
---
# Configuration Reference

## Environment

The application runs in the `production` environment by default.

The environment is selected with the `APP_ENV` variable.

Valid environment values are `development`, `staging`, and `production`.

## Server Configuration

The application listens on port `8080`.

The server request timeout is `30` seconds.

The maximum request body size is `10 MB`.

## Database Configuration

The database connection pool has a maximum size of `20` connections.

The database connection timeout is `5` seconds.

## Logging

The default log level is `INFO`.

The application writes logs in JSON format.

## Feature Flags

The new dashboard is disabled by default.

The feature flag for the new dashboard is `ENABLE_NEW_DASHBOARD`.