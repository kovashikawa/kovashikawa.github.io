# Rafael Kovashikawa — Personal Site

This repository contains the source for **[kovashikawa.github.io](https://kovashikawa.github.io)**, my portfolio and blog.  
It runs on the excellent **[Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes)** Jekyll theme with a few custom tweaks.

## What’s inside

| Directory / file | Purpose |
| ---------------- | ------- |
| `_config.yml` | Global site configuration |
| `_data/navigation.yml` | Top‑level navigation links |
| `_pages/` | Stand‑alone pages (About, Projects, etc.) |
| `_posts/` | Blog articles |
| `_sass/_custom.scss` | Theme overrides (colors, fonts, layout) |
| `assets/images/` | Images (avatar, banners, screenshots) |

## Local development

```bash
# one‑time setup
bundle install

# live preview at http://localhost:4000
bundle exec jekyll serve
```

## Deployment

Pushing to **`main`** triggers an automatic GitHub Pages build and deploy.

## House‑cleaning notes

The upstream Minimal Mistakes README shipped with sections about skins, demo pages, screenshots, and installation instructions for the theme itself.  
Those are no longer relevant to *this* repo and have been removed. Files that can also be safely ignored or deleted:

* `screenshot.png`, `screenshot-layouts.png`
* `CHANGELOG.md`, original `docs/` demo content

## License

* Theme: MIT (see <https://github.com/mmistakes/minimal-mistakes>).  
* Site content: © 2025 Rafael Kovashikawa — all rights reserved unless noted otherwise.