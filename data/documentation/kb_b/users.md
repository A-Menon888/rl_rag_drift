# Users API

The Users API manages application user accounts.
Create a user with POST /v2/users.
Retrieve a user with GET /v2/users/{id}.
Update a user with PATCH /v2/users/{id}.
Delete a user with DELETE /v2/users/{id}.
The user identifier is an integer.
The email field is required when creating a user.
Email addresses must be unique.
The name field is optional.
The default account status is pending.
New users receive a unique identifier.
A successful creation returns HTTP 201.
A successful retrieval returns HTTP 200.
Missing users return HTTP 404.
Invalid request data returns HTTP 400.
Deleted users cannot access the application.
User deletion is permanent.
The API returns JSON responses.
Passwords are never returned in user responses.
Email addresses are stored in lowercase.
User creation requires authentication.
User updates require authentication.
Administrators can delete any user.
Regular users can update their own profile.
The API supports pagination for user listings.
The default page size is 50 users.
The maximum page size is 200 users.
User records contain created_at timestamps.
User records contain updated_at timestamps.