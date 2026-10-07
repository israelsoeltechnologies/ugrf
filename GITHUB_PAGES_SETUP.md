# Publish the UGRF documentation with GitHub Pages

These files are prepared for:

```text
israelsoeltechnologies/ugrf
```

## 1. Copy the files

Copy:

```text
mkdocs.yml
.github/workflows/docs.yml
docs/
```

into the matching paths in the repository.

The existing `docs/model_spec.md` is intentionally replaced by the expanded version.

## 2. Commit and push

```bash
git add mkdocs.yml docs .github/workflows/docs.yml
git commit -m "Add UGRF documentation site"
git push
```

## 3. Enable GitHub Pages

In GitHub:

```text
Settings
→ Pages
→ Build and deployment
→ Source
→ GitHub Actions
```

## 4. Expected public URL

```text
https://israelsoeltechnologies.github.io/ugrf/
```

## 5. Update package metadata after the site is live

Change the existing `pyproject.toml` Documentation URL to:

```toml
Documentation = "https://israelsoeltechnologies.github.io/ugrf/"
```

## 6. Local preview

```bash
pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

For a strict build:

```bash
mkdocs build --strict
```

## Important API note

These docs describe the intended v1 update where `model.predict()` returns the lead-by-lead point forecast path and `model.predict_cumulative()` returns the cumulative horizon forecast. Apply the pending point-forecast API update before treating this documentation as matching the released PyPI artifact.
