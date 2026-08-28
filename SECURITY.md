# Security Policy

## Supported Versions

Opifex is a research preview. Security fixes target the latest released version
and the current `main` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately by emailing
security@avitai.bio.

Include:

- Affected Opifex version or commit.
- Environment details, including Python, JAX, and operating system versions.
- A minimal reproduction or proof of impact.
- Any known mitigations.

We will acknowledge reports as soon as practical, investigate privately, and
coordinate disclosure once a fix or mitigation is available.

## Scope

Security-sensitive areas include checkpoint and weight loading, dataset connectors, experiment-tracker integrations, quantum-chemistry input parsing, file-system storage, CI release automation, and any code path that processes untrusted model or data artifacts.
