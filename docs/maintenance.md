# Maintenance

Use `main` for reviewed changes and version tags to identify immutable releases. GitHub Actions runs the scope audit, syntax checks, CPU tests, and synthetic examples. Dependabot checks workflow actions monthly. Maintainers review issues, dependency updates, and release changes in a maintenance session.

For reproducibility reports, record the tag, input schema, software versions, exact command, and sanitized output. Keep original passages, private answers, weights, and manuscript material out of Git branches, release attachments, issue text, and Actions artifacts. Use `python scripts/audit_release.py` against the staged tree before a push. Verify the tag and its reachable history before publication.

Record changes to implementation or metrics in the changelog. Publish performance results only with a data and model provenance record, a fixed protocol, and a documented rerun. Preserve earlier tags for comparison.
