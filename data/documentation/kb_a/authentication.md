# Authentication

The API uses token-based authentication.
Clients create tokens through the authentication endpoint.
The token endpoint is POST /auth/token.
Clients provide username and password to create a token.
A successful request returns an access token.
Access tokens expire after 3600 seconds.
Requests include the token in the Authorization header.
The required format is Bearer <token>.
Invalid credentials return HTTP 401.
Expired tokens also return HTTP 401.
Tokens should not be stored in application logs.
The API supports HTTPS for all authentication requests.
Tokens are scoped to the application.
A token can be revoked by the administrator.
Revoked tokens cannot be used for API requests.
The authentication service runs on port 8000.
The token endpoint accepts JSON requests.
The username field is required.
The password field is required.
Missing credentials return HTTP 400.
Clients should refresh tokens before expiration.
The refresh endpoint is POST /auth/refresh.
Refresh tokens are valid for 24 hours.
The API limits authentication attempts to 10 per minute.
Repeated failures may temporarily block an account.
Successful authentication records an audit event.
Audit events contain the user identifier.
Authentication errors should not expose passwords.
Service-to-service requests use the same token format.