# VitalDB provenance and usage notes

Planned default integration path:
1. `vitaldb` Python library (preferred)
2. VitalDB Web API fallback

For each ingest run, HALO must record:
- source type (library/api)
- requested case IDs
- requested tracks
- fetch timestamp
- config hash and seed

Before broad usage, maintainers should verify and document compliance with VitalDB terms in this file.
