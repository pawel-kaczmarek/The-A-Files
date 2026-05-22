# Publishing

These steps prepare distributions only. Do not upload unless the release version and tag are intentional.

## Local build

Install development tools:

```bash
python -m pip install -e .[dev]
```

Build the source distribution and wheel:

```bash
python -m build
```

Validate distribution metadata:

```bash
python -m twine check dist/*
```

## TestPyPI

Upload to TestPyPI with an API token:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ the-a-files
```

## PyPI Release

The GitHub Actions workflow publishes to PyPI through Trusted Publishing when a version tag is pushed:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Before creating the tag, make sure `pyproject.toml` has the intended version and that GitHub has a PyPI trusted publisher configured for:

- repository: `pawel-kaczmarek/The-A-Files`
- workflow: `ci.yml`
- environment: `pypi`

## Vendored SRMRpy

`SRMRpy` is not published on PyPI, and PyPI-uploaded distributions cannot rely on direct URL dependencies. The pinned SRMRpy source is vendored under `src/taf/metrics/speech_reverberation/srmrpy/` with its MIT license notice in `LICENSES/SRMRpy-LICENSE.md`.

SRMRpy depends on `gammatone`, which is available on PyPI and is declared as a normal pinned dependency.
