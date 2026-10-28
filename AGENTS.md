# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

VuePress 2 documentation site using `vuepress-theme-hope` theme. Bilingual content (English and Chinese).

## Commands

- `pnpm docs:dev` — Start local development server
- `pnpm docs:build` — Production build (outputs to `src/.vuepress/dist/`)
- `pnpm docs:clean-dev` — Dev server with clean cache (use when config changes don't take effect)
- `pnpm docs:update-package` — Update VuePress packages

Package manager: pnpm. ESM project (`"type": "module"`).

## Architecture

Content lives in `src/` with two locale trees:
- `src/` — English content
- `src/zh/` — Chinese content (mirrors English structure)

VuePress configuration in `src/.vuepress/`:
- `config.ts` — Main VuePress config (locales, base path)
- `theme.ts` — Theme configuration via `hopeTheme()` (navbar, sidebar, plugins, markdown features, encryption)
- `navbar/` — Navigation bar definitions per locale (`en.ts`, `zh.ts`, `index.ts`)
- `sidebar/` — Sidebar definitions per locale (`en.ts`, `zh.ts`, `index.ts`)
- `styles/` — SCSS overrides (`config.scss` sets theme color `#096dd9`)
- `public/` — Static assets (favicon, logos, images)

## Content Structure

Markdown files use VuePress frontmatter. Key conventions:
- `README.md` in a directory becomes the index page for that route
- Bilingual content must be added to both `src/` and `src/zh/` trees
- Navbar/sidebar configs in `src/.vuepress/navbar/` and `src/.vuepress/sidebar/` must be updated when adding new sections

## CI/CD

GitHub Actions workflow `deploy-docs.yml`:
- Triggers on push to `main`
- Builds with Node 24 and pnpm
- Deploys to `gh-pages` branch
