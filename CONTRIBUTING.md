# Contributing

## Code Quality

Pre-commit hooks are required:

```bash
pre-commit run --all-files
```

Checks include:

* formatting (`end-of-file-fixer`)
* YAML validation
* large file prevention
* linting (`ruff`)
* type checking (`mypy`)

All checks must pass before submitting changes.

---

## Tests

Run tests with:

```bash
pytest
```

---

## Commits

Use clear, structured messages:

* `Add: ...`
* `Fix: ...`
* `Test: ...`

Use the imperative form and keep messages concise.

---

## Pull Requests

* Work on a separate branch
* Keep changes focused
* Ensure tests and pre-commit checks pass
* Provide a clear description of the change

---

## Changelog

Document user-visible changes in `CHANGELOG.md` under the appropriate version.

---

## Git Hygiene

* Do not add personal or machine-specific files to `.gitignore`
* Use `.git/info/exclude` for local files
* Do not commit data, caches, or temporary files
