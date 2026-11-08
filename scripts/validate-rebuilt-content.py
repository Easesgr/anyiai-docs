"""Validate all integrated content and navigation without creating site builds."""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


SPEC = importlib.util.spec_from_file_location('rebuild_content', Path(__file__).with_name('rebuild-content.py'))
REBUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REBUILD)


def validate() -> dict:
    pages = REBUILD.read_json(REBUILD.OUT / 'inventory.json')['pages']
    branding_policy = REBUILD.load_policy(REBUILD.ROOT)
    branding = REBUILD.BrandingStore(REBUILD.ROOT, branding_policy, native_root=REBUILD.OUT) if branding_policy else None
    available = {page['path'] for page in pages}
    issues = []
    counts = Counter(pages=len(pages))
    for page in pages:
        folder = REBUILD.page_folder(page)
        target = REBUILD.SRC / page['path']
        if not target.is_file():
            issues.append({'path': page['path'], 'problem': 'Source page is missing'})
            continue
        article = REBUILD.read_json(folder / 'reviewed.json')
        current_hash = REBUILD.digest(article['body_markdown'])
        audit = folder / 'final-audit.json'
        if not audit.is_file() or REBUILD.read_json(audit).get('body_sha256') != current_hash:
            issues.append({'path': page['path'], 'problem': 'Final factual audit is missing or stale'})
        snapshot = REBUILD.media_snapshot(page)
        if not REBUILD.media_ready(page, snapshot):
            issues.append({'path': page['path'], 'problem': 'Current native images or required visual review are missing'})
        record = folder / 'integrated.json'
        integrated = REBUILD.read_json(record) if record.is_file() else {}
        if integrated.get('content_sha256') != REBUILD.digest(target.read_text()):
            issues.append({'path': page['path'], 'problem': 'Source content differs from integrated record'})
        if record.is_file() and integrated.get('media_snapshot') != snapshot:
            issues.append({'path': page['path'], 'problem': 'Integrated article or image versions are stale'})
        front, body = REBUILD.split_document(target.read_text())
        if front.get('title') != article['title'] or front.get('description') != article['description']:
            issues.append({'path': page['path'], 'problem': 'Page metadata differs from final article'})
        if 'rebuild:' in body:
            issues.append({'path': page['path'], 'problem': 'Unresolved illustration anchor'})
        if page['path'] == 'README.md' and '<HomePortal' not in body:
            issues.append({'path': page['path'], 'problem': 'Homepage portal is missing'})
        counts['new_lessons'] += bool(page.get('new'))
        prose = re.sub(r'^```[^\n]*\n.*?^```\s*$', '', body, flags=re.S | re.M)
        for image in article['images']:
            url = f"/images/content-rebuild/{page['id']}/{image['id']}.png"
            problem = REBUILD.published_image_problem(
                page, image, branding, integrated.get('published_images', {}).get(image['id']))
            if problem:
                issues.append({'path': page['path'], 'problem': problem})
            if body.count('](' + url + ')') != 1:
                issues.append({'path': page['path'], 'problem': 'Illustration must appear exactly once: ' + image['id']})
            counts['images'] += 1
        for match in re.finditer(r'(?<!!)\[[^\]\n]+\]\(([^)\s]+)\)', prose):
            url = match[1]
            if url.startswith(('#', 'http:', 'https:', 'mailto:', 'tel:')):
                continue
            path = unquote(urlsplit(url).path)
            path = path.lstrip('/') if path.startswith('/') else str(Path(page['path']).parent / path)
            if not path or path.endswith('/'):
                path += 'README.md'
            path = re.sub(r'\.html$', '.md', path)
            if path not in available:
                issues.append({'path': page['path'], 'problem': 'Broken internal link: ' + url})
    navigation = REBUILD.OUT / 'navigation.json'
    if not navigation.is_file():
        issues.append({'path': '.vuepress/sidebar/zh.ts', 'problem': 'Rebuilt navigation is missing'})
    else:
        links = set()

        def visit(value):
            if isinstance(value, dict):
                if 'link' in value:
                    links.add(value['link'])
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(REBUILD.read_json(navigation))
        for page in pages:
            if page['path'] != 'README.md' and '/' + page['path'].replace('README.md', '') not in links:
                issues.append({'path': page['path'], 'problem': 'Page is absent from the rebuilt sidebar'})
    portal_path = REBUILD.SRC / '.vuepress/components/HomePortal.vue'
    for url in set(re.findall(r'["\'](/images/[^"\']+)["\']', portal_path.read_text())):
        if not url.startswith('/images/content-rebuild/') and 'qrcode' not in url:
            issues.append({'path': str(portal_path.relative_to(REBUILD.SRC)),
                           'problem': 'Homepage still references an old teaching image: ' + url})
        if not (REBUILD.SRC / '.vuepress/public' / url.lstrip('/')).is_file():
            issues.append({'path': str(portal_path.relative_to(REBUILD.SRC)),
                           'problem': 'Missing homepage image: ' + url})
    preview_manifest = REBUILD.SRC / '.vuepress/public/images/home-previews/manifest.json'
    if preview_manifest.is_file():
        try:
            counts['home_previews'] = REBUILD.verify_home_previews(REBUILD.ROOT, branding_policy)['images']
        except (REBUILD.BrandingError, OSError, ValueError) as error:
            issues.append({'path': str(preview_manifest.relative_to(REBUILD.SRC)), 'problem': str(error)})
    result = {'passed': not issues, 'counts': dict(counts), 'issues': issues,
              'visual_review': 'required' if REBUILD.visual_review_required() else 'skipped_by_user',
              'image_branding': branding_policy.fingerprint if branding_policy else None}
    REBUILD.write_json(REBUILD.OUT / 'validation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if validate()['passed'] else 1)
