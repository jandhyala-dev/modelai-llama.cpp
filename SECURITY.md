# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in modelai-llama.cpp, please report it responsibly.

**Do NOT open a public issue.**

Instead, use [GitHub's private vulnerability reporting](https://github.com/jandhyala-dev/modelai-llama.cpp/security/advisories/new) to submit your report. This ensures the vulnerability is handled privately until a fix is available.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions or commits
- Potential impact (crash, data corruption, remote code execution, etc.)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 7 days
- **Fix or mitigation:** Depends on severity, targeting 30 days for Critical/High

### Severity Levels

| Severity | CVSS | Examples |
|----------|------|---------|
| Critical | >= 9.0 | Remote code execution, full system compromise |
| High | 7.0 - 8.9 | RCE in limited contexts, significant data loss |
| Moderate | 4.0 - 6.9 | Denial of service, partial disruption |
| Low | < 4.0 | Information disclosure, non-exploitable flaws |

### Track Record

- **RPC RCE patch (#20908):** Synced same-day from upstream llama.cpp and deployed immediately. See [UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) for details.

## Supported Versions

Only the latest release on `modelai-main` is supported with security updates.
