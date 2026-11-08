#!/usr/bin/env python3
"""Resize the homepage's existing course illustrations without modifying their originals."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

from image_branding import BrandingError, BrandingStore, atomic_bytes, atomic_json, load_policy, verify_home_previews

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / 'src/.vuepress/public'
COMPONENT = ROOT / 'src/.vuepress/components/HomePortal.vue'
STAGES = ROOT / 'src/.vuepress/data/learning.ts'
DESTINATION = PUBLIC / 'images/home-previews'
RECIPE = 'Pillow contain, LANCZOS, WebP quality 84, method 6; aspect ratio preserved.'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    policy = load_policy(ROOT)
    branding = BrandingStore(ROOT, policy) if policy else None
    stage_ids = set(re.findall(r'content-rebuild/([a-f0-9]{16})/', STAGES.read_text()))
    cover_ids = stage_ids | set(re.findall(r"'([a-f0-9]{16})'", COMPONENT.read_text()))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    previous_images = {row['preview']: row for row in previous.get('images', [])}
    previous_manifest_hash = sha256(manifest_path) if manifest_path.is_file() else None
    images = []
    prepared = []
    for cover_id in sorted(cover_ids):
        source = PUBLIC / f'images/content-rebuild/{cover_id}/cover.png'
        target = DESTINATION / f'{cover_id}.webp'
        source_branding = None
        if branding:
            record = branding.inspect(source.relative_to(PUBLIC / 'images').as_posix(), require_published=True)
            source_branding = branding.proof(record)
        source_bytes = source.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if source_branding and source_branding['published_sha256'] != source_hash:
            raise BrandingError('Homepage source changed during preview preparation: ' + str(source))
        width = 960 if cover_id in stage_ids else 640
        with Image.open(io.BytesIO(source_bytes)) as image:
            original_size = list(image.size)
            thumbnail = ImageOps.contain(image.convert('RGB'), (width, width), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            thumbnail.save(buffer, 'WEBP', quality=84, method=6)
            preview_size = list(thumbnail.size)
        payload = buffer.getvalue()
        preview_hash = hashlib.sha256(payload).hexdigest()
        preview_url = '/' + str(target.relative_to(PUBLIC))
        before_hash = sha256(target) if target.is_file() else None
        # A prior interrupted run may have already written these exact bytes.
        allowed = {preview_hash, previous_images.get(preview_url, {}).get('previewSha256')}
        if before_hash is not None and before_hash not in allowed:
            raise BrandingError('Homepage preview changed outside its manifest: ' + str(target))
        images.append({
            'source': '/' + str(source.relative_to(PUBLIC)),
            'sourceSha256': source_hash,
            'sourceDimensions': original_size,
            'sourceBytes': len(source_bytes),
            'preview': preview_url,
            'previewSha256': preview_hash,
            'previewDimensions': preview_size,
            'previewBytes': len(payload),
            **({'sourceBranding': source_branding} if source_branding else {}),
        })
        prepared.append((source, source_hash, target, before_hash, preview_hash, payload))
    manifest = {
        'provenance': 'Resized existing marker illustrations; no new image generation.',
        'recipe': RECIPE,
        'images': images,
        **({'brandingPolicySha256': policy.fingerprint} if policy else {}),
    }
    for source, source_hash, target, before_hash, preview_hash, payload in prepared:
        if policy:
            policy.assert_current()
        if sha256(source) != source_hash or (sha256(target) if target.is_file() else None) != before_hash:
            raise BrandingError('Image changed while preparing homepage previews: ' + str(target))
        if before_hash != preview_hash:
            atomic_bytes(target, payload)
    if (sha256(manifest_path) if manifest_path.is_file() else None) != previous_manifest_hash:
        raise BrandingError('Homepage preview manifest changed during preparation')
    if any(sha256(source) != source_hash or sha256(target) != preview_hash
           for source, source_hash, target, _, preview_hash, _ in prepared):
        raise BrandingError('Homepage source or preview changed before manifest publication')
    if policy:
        policy.assert_current()
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        'images': len(images),
        'sourceBytes': sum(item['sourceBytes'] for item in images),
        'previewBytes': sum(item['previewBytes'] for item in images),
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Only verify recorded source/output hashes and branding')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_home_previews(ROOT, load_policy(ROOT))))
    else:
        main()
