# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Chinese VuePress 2 documentation site using `vuepress-theme-hope`, Vue 3 and Vite. Content, navigation and shared styles live in this repository; the public site uses Tencent Cloud COS/CDN.

## Commands

- `pnpm docs:dev` — Start local development server
- `pnpm docs:build` — Production build (outputs to `src/.vuepress/dist/`)
- `pnpm docs:clean-dev` — Dev server with clean cache (use when config changes don't take effect)
- `pnpm docs:update-package` — Update VuePress packages

Package manager: pnpm. ESM project (`"type": "module"`).

## Architecture

Chinese content lives in `src/`, organized into fundamentals, llm-100, interview, frontier, learning-path, projects, architecture, resources and about.

VuePress configuration in `src/.vuepress/`:
- `config.ts` — Main VuePress config (locales, base path)
- `theme.ts` — Theme configuration via `hopeTheme()` (navbar, sidebar, plugins, markdown features, encryption)
- `navbar/` — Navigation bar definitions per locale (`en.ts`, `zh.ts`, `index.ts`)
- `sidebar/` — Sidebar definitions per locale (`en.ts`, `zh.ts`, `index.ts`)
- `styles/` — Shared typography, navigation, responsive layouts and green theme
- `components/`, `layouts/`, `data/` — Homepage, reading layout and course routes
- `public/` — Static assets (favicon, logos, images)

## Content Structure

Markdown files use VuePress frontmatter. Key conventions:
- `README.md` in a directory becomes the index page for that route
- Navbar/sidebar configs in `src/.vuepress/navbar/` and `src/.vuepress/sidebar/` must be updated when adding new sections
- Full-resolution illustrations use Git LFS. Run `git lfs pull` before a deployable build. Keep images, text, paths and real author/QR information consistent.
- Do not commit plaintext reading codes, credentials, local model configuration, browser artifacts or generated `output/` records.

## CI/CD

GitHub Actions workflow `deploy-docs.yml`:
- Triggers on push to `main`
- Checks Git LFS pointers and page compilation with Node 24 and pinned pnpm
- Push checks do not download full-resolution images or publish their build output
- GitHub Pages publishing is manual and checks the output size; the complete site is deployed to COS/CDN
