# Documentation and GitHub Pages

The site uses MkDocs and the free, open-source Material theme. It requires no paid theme, analytics
service or external font service. Python documentation dependencies are separate from the audio package.

## Local preview

From the repository root:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Open **http://127.0.0.1:8000**. If the research API is already using that port:

```bash
python -m mkdocs serve --dev-addr 127.0.0.1:8001
```

Build the deployable static site and validate links:

```bash
python -m mkdocs build --strict
```

Output goes to `site/`, which is ignored by Git. Edit Markdown in `docs/`, navigation in `mkdocs.yml`
and the small visual overrides in `docs/stylesheets/extra.css`.

## Publish

1. In the repository’s **Settings → Pages → Build and deployment**, select **GitHub Actions** as the source.
2. Push the documentation changes to the default branch (`master` in this checkout).
3. The **Documentation** workflow validates and builds the site, uploads the Pages artifact and deploys
   it using the `github-pages` environment. Pull requests build without deploying.
4. The expected URL is **https://pawelkaczmarek12.github.io/the-a-files/**.

The workflow can also be started manually from the Actions tab. Publication requires repository access
and an enabled Pages environment. If the owner or repository name changes, update `site_url`, `repo_url`
and the README documentation link before publishing.

See [GitHub’s custom-workflow instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [Material’s publication guide](https://squidfunk.github.io/mkdocs-material/publishing-your-site/).

## Maintaining scientific descriptions

Check catalogue coverage against `src/taf/methods/factory.py`, `src/taf/metrics/factory.py` and
`src/taf/attacks/registry.py`. For each addition, document the mechanism, identifier, limitations and source.
Distinguish paper algorithms, local adaptations and wrappers over released models. Cite a paper’s
section or experimental conditions when quoting results; do not imply those results were reproduced locally.
