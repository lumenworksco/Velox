# Security Policy

## Secrets Management

- **Never commit API keys, secrets, or `.env` files** to the repository.
- Use environment variables for all sensitive configuration:
  - `ALPACA_API_KEY`
  - `ALPACA_SECRET_KEY`
  - `TELEGRAM_TOKEN`
  - `TELEGRAM_CHAT_ID`
- The `.gitignore` file excludes `.env`, `state.json`, and database files by default.
- If using Docker, pass secrets via environment variables or Docker secrets -- do not bake them into images.

## Reporting Security Issues

If you discover a security vulnerability, **do not open a public issue**.

Instead, please report it privately via email:

**security@example.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Best Practices

- Run in paper trading mode (`ALPACA_LIVE=false`) until you have thoroughly tested your configuration.
- Use a dedicated API key with minimal permissions for the bot.
- Rotate API keys periodically.
- Monitor the bot's activity and set up alerts for unexpected behavior.
- Keep dependencies up to date to patch known vulnerabilities.
