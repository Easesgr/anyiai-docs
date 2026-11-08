"""Verified, resumable publication derivatives; provider originals remain immutable.

The per-path records are the transaction journal. A prepared record is durable
before a public file is replaced, so either side of an interrupted replace can
be identified by its digest. manifest.json is only a convenient inventory.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION


ENGINE = 'anyi-image-branding-v2-overlay'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
DEFAULT_PLACEMENT = {
    'width_fraction': 0.12, 'margin_fraction': 0.012,
}


class BrandingError(ValueError):
    """A provenance, policy, or concurrent-write mismatch; never overwrite it."""


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=24000)
def _file_hash(path, version):
    with Path(path).open('rb') as stream:
        result = hashlib.file_digest(stream, 'sha256').hexdigest()
    if identity(Path(path)) != version:
        raise BrandingError('File changed while hashing: ' + path)
    return result


def file_hash(path):
    return _file_hash(str(path), identity(path))


def read_json(path):
    return json.loads(path.read_text())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, payload, *, immutable=False, guard=None):
    """Flush bytes, atomically publish, and flush the directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            temporary.chmod(0o444)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if file_hash(path) != hashlib.sha256(payload).hexdigest():
                    raise BrandingError('Immutable backup differs: ' + str(path))
        else:
            if guard:
                guard()
            os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value, *, immutable=False):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode(),
                 immutable=immutable)


def relative_path(root, path):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise BrandingError('Path escapes the repository: ' + str(path)) from error


def rooted(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise BrandingError('Expected a repository-relative path: ' + repr(value))
    path = root / value
    relative_path(root, path)
    return path


class BrandingPolicy:
    def __init__(self, root, path, config):
        self.root = Path(root).resolve()
        self.path = path
        self.config_hash = file_hash(path)
        allowed = {'version', 'enabled', 'logo', 'logo_version', 'placement', 'exclude'}
        if set(config) - allowed or config.get('version') != 1 or config.get('enabled') is not True:
            raise BrandingError('Unknown or invalid image-branding policy fields')
        placement = config.get('placement', {})
        if not isinstance(placement, dict) or set(placement) - set(DEFAULT_PLACEMENT):
            raise BrandingError('Unknown image-branding placement fields')
        self.placement = {**DEFAULT_PLACEMENT, **placement}
        for field, low, high in [('width_fraction', 0.03, 0.3), ('margin_fraction', 0.001, 0.05)]:
            value = self.placement[field]
            if type(value) not in (float, int) or not low <= value <= high:
                raise BrandingError('Invalid placement ' + field)
        self.excludes = config.get('exclude', [])
        if not isinstance(self.excludes, list) or any(not isinstance(item, str) for item in self.excludes):
            raise BrandingError('exclude must be an array of image-relative glob patterns')
        self.logo_path = rooted(self.root, config.get('logo'))
        self.logo_version = config.get('logo_version')
        if not isinstance(self.logo_version, str) or not self.logo_version.strip():
            raise BrandingError('logo_version is required')
        logo_bytes = self.logo_path.read_bytes()
        self.logo_hash = hashlib.sha256(logo_bytes).hexdigest()
        with Image.open(io.BytesIO(logo_bytes)) as image:
            if image.mode != 'RGBA' or image.getextrema()[3][0] == 255:
                raise BrandingError('Logo must be the existing transparent RGBA raster')
            self.logo_bbox = image.getchannel('A').getbbox()
            if self.logo_bbox is None:
                raise BrandingError('Logo is entirely transparent')
            self.logo = image.crop(self.logo_bbox).copy()
        self.provenance = {
            'engine': ENGINE, 'pillow': PILLOW_VERSION, 'logo': config['logo'],
            'logo_sha256': self.logo_hash, 'logo_version': self.logo_version,
            'logo_alpha_bbox': list(self.logo_bbox), 'placement': self.placement,
            'exclude': self.excludes,
            'composition': 'Transparent logo over original bottom-right pixels; no added canvas or strip',
            'encoding': 'PNG lossless; WebP lossless exact method 6; JPEG quality 98 subsampling 0',
        }
        self.fingerprint = value_hash(self.provenance)

    def assert_current(self):
        if file_hash(self.path) != self.config_hash or file_hash(self.logo_path) != self.logo_hash:
            raise BrandingError('Branding policy or logo changed during this operation')

    def excluded(self, path):
        # Derived previews must always be regenerated from branded sources.
        return (path.startswith(('branding/', 'home-previews/'))
                or any(fnmatchcase(path, pattern) for pattern in self.excludes))


def load_policy(root, config_path=None):
    root = Path(root).resolve()
    path = Path(config_path) if config_path else root / 'config/image-branding.json'
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if config_path:
            raise BrandingError('Branding config is missing: ' + str(path))
        return None
    config = read_json(path)
    if not isinstance(config, dict):
        raise BrandingError('Branding config must be an object')
    if config.get('enabled') is False:
        return None
    return BrandingPolicy(root, path, config)


def render(source_bytes, policy):
    """Composite the transparent logo directly onto the original image canvas."""
    policy.assert_current()
    with Image.open(io.BytesIO(source_bytes)) as source:
        if getattr(source, 'n_frames', 1) != 1:
            raise BrandingError('Animated teaching images require an explicit publication rule')
        source_format = source.format
        if source_format not in {'PNG', 'JPEG', 'WEBP'}:
            raise BrandingError('Unsupported raster encoding: ' + str(source_format))
        stored_size = list(source.size)
        orientation = source.getexif().get(274, 1)
        info = {key: source.info[key] for key in ('icc_profile', 'dpi') if key in source.info}
        image = ImageOps.exif_transpose(source).convert('RGBA')
    width, height = image.size
    margin = min(max(1, round(min(width, height) * policy.placement['margin_fraction'])), max(0, (width - 1) // 2))
    logo_width = max(1, min(round(width * policy.placement['width_fraction']), width - 2 * margin))
    available_height = max(1, height - 2 * margin)
    logo_width = min(logo_width, max(1, int(available_height * policy.logo.width / policy.logo.height)))
    logo_height = max(1, round(logo_width * policy.logo.height / policy.logo.width))
    logo = policy.logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
    x, y = width - margin - logo_width, height - margin - logo_height
    canvas = image.copy()
    canvas.alpha_composite(logo, (x, y))
    if image.getchannel('A').getextrema() == (255, 255):
        canvas = canvas.convert('RGB')
    stream = io.BytesIO()
    if source_format == 'PNG':
        canvas.save(stream, 'PNG', compress_level=6, **info)
    elif source_format == 'WEBP':
        canvas.save(stream, 'WEBP', lossless=True, exact=True, method=6,
                    **{key: value for key, value in info.items() if key == 'icc_profile'})
    else:
        canvas.convert('RGB').save(stream, 'JPEG', quality=98, subsampling=0, **info)
    details = {
        'original_dimensions': stored_size, 'oriented_dimensions': [width, height],
        'published_dimensions': list(canvas.size), 'source_format': source_format,
        'published_format': source_format, 'orientation_normalized': orientation != 1,
        'method': 'bottom-right-overlay', 'logo_bbox': [x, y, x + logo_width, y + logo_height],
    }
    return stream.getvalue(), details


class BrandingStore:
    def __init__(self, root, policy, *, native_root=None):
        self.root = Path(root).resolve()
        self.policy = policy
        self.public = self.root / 'src/.vuepress/public/images'
        self.state = self.root / 'output/image-branding'
        self.native_root = Path(native_root) if native_root else self.root / 'output/content-rebuild'

    def key(self, relative):
        return hashlib.sha256(relative.encode()).hexdigest()

    def path(self, relative):
        target = rooted(self.public, relative)
        if target.suffix.lower() not in EXTENSIONS or self.policy.excluded(relative):
            raise BrandingError('Path is not an eligible teaching image: ' + relative)
        if target.is_symlink():
            raise BrandingError('Refusing to replace a symbolic-link image: ' + relative)
        return target

    def record_path(self, relative):
        return self.state / 'records' / (self.key(relative) + '.json')

    @contextmanager
    def lock(self, relative):
        path = self.state / 'locks' / (self.key(relative) + '.lock')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def native_for(self, relative):
        parts = Path(relative).parts
        if parts[0] != 'content-rebuild':
            return None
        if len(parts) != 3:
            raise BrandingError('Unexpected content-rebuild image path: ' + relative)
        return self.native_root / 'pages' / parts[1] / 'images' / parts[2]

    def inventory(self, only=None):
        if only:
            paths = sorted(set(only))
            for relative in paths:
                if not self.path(relative).is_file():
                    raise BrandingError('Teaching image is missing: ' + relative)
            return paths
        return sorted(path.relative_to(self.public).as_posix() for path in self.public.rglob('*')
                      if path.is_file() and path.suffix.lower() in EXTENSIONS
                      and not self.policy.excluded(path.relative_to(self.public).as_posix()))

    def load_record(self, relative, *, allow_policy_update=False):
        path = self.record_path(relative)
        if not path.is_file():
            return None
        record = read_json(path)
        if (not isinstance(record, dict) or record.get('path') != relative or record.get('version') != 1
                or record.get('state') not in {'prepared', 'published'}
                or record.get('record_sha256') != value_hash({key: value for key, value in record.items()
                                                             if key != 'record_sha256'})):
            raise BrandingError('Invalid branding provenance record: ' + relative)
        if (not isinstance(record.get('policy'), dict)
                or record.get('policy_sha256') != value_hash(record['policy'])):
            raise BrandingError('Invalid branding policy provenance: ' + relative)
        if (not allow_policy_update and (record.get('policy_sha256') != self.policy.fingerprint
                                        or record.get('policy') != self.policy.provenance)):
            raise BrandingError('Stale branding policy or logo: ' + relative)
        original_hash = record.get('original_sha256', '')
        if (len(original_hash) != 64 or any(char not in '0123456789abcdef' for char in original_hash)
                or record.get('backup') != 'originals/' + original_hash):
            raise BrandingError('Invalid original backup provenance: ' + relative)
        backup = self.state / record['backup']
        if not backup.is_file() or file_hash(backup) != original_hash:
            raise BrandingError('Original backup is missing or changed: ' + relative)
        for field in ('published_sha256', 'original_dimensions', 'published_dimensions', 'logo_bbox', 'method'):
            if field not in record:
                raise BrandingError('Incomplete branding provenance: ' + relative)
        return record

    def save_record(self, relative, record):
        record = {key: value for key, value in record.items() if key != 'record_sha256'}
        record['record_sha256'] = value_hash(record)
        atomic_json(self.record_path(relative), record)
        return record

    def check_native(self, relative, source_hash, native=None):
        native = native if native is not None else self.native_for(relative)
        if native is not None and (not native.is_file() or file_hash(native) != source_hash):
            raise BrandingError('Native source is missing or changed: ' + relative)

    def inspect(self, relative, *, deep=False, require_published=False, native=None):
        self.policy.assert_current()
        public = self.path(relative)
        if not public.is_file():
            raise BrandingError('Published image is missing: ' + relative)
        current = file_hash(public)
        record = self.load_record(relative)
        if record is None:
            if require_published:
                raise BrandingError('Branding provenance is missing: ' + relative)
            self.check_native(relative, current, native)
            with Image.open(public) as image:
                dimensions = list(image.size)
            return {'path': relative, 'status': 'pending', 'original_sha256': current,
                    'original_dimensions': dimensions, 'original_bytes': public.stat().st_size}
        self.check_native(relative, record['original_sha256'], native)
        if current == record['published_sha256']:
            state = 'published' if record['state'] == 'published' else 'recoverable'
        elif record['state'] == 'prepared' and current == record['expected_public_sha256']:
            state = 'prepared'
        else:
            raise BrandingError('Published image changed outside the branding transaction: ' + relative)
        if require_published and state != 'published':
            raise BrandingError('Branding transaction needs apply to resume: ' + relative)
        if deep:
            payload, details = render((self.state / record['backup']).read_bytes(), self.policy)
            if (hashlib.sha256(payload).hexdigest() != record['published_sha256']
                    or any(record.get(key) != value for key, value in details.items())):
                raise BrandingError('Branded output cannot be reproduced from its original: ' + relative)
        return {**record, 'status': state}

    def proof(self, record):
        return {key: record[key] for key in ('path', 'original_sha256', 'published_sha256', 'policy_sha256')}

    def publish(self, relative, *, source=None, expected_source_hash=None,
                allow_source_update=False, expected_existing_hash=None, guard=None, allow_policy_update=False):
        """Commit one image. Only integration may supply a new, receipt-verified native."""
        with self.lock(relative):
            self.policy.assert_current()
            public = self.path(relative)
            before_version = identity(public) if public.exists() else None
            before_hash = file_hash(public) if public.exists() else None
            record_path = self.record_path(relative)
            record_version = identity(record_path) if record_path.exists() else None
            record = self.load_record(relative, allow_policy_update=allow_policy_update)
            native = source if source is not None else self.native_for(relative)
            if record:
                policy_changed = record['policy_sha256'] != self.policy.fingerprint
                if policy_changed and record['state'] != 'published':
                    raise BrandingError('Resume the previous policy transaction before changing its policy: ' + relative)
                if before_hash not in ({record['published_sha256']} if record['state'] == 'published'
                                       else {record['published_sha256'], record['expected_public_sha256']}):
                    raise BrandingError('Published image changed outside the branding transaction: ' + relative)
                native_hash = file_hash(native) if native is not None and native.is_file() else None
                source_changed = native is not None and native_hash != record['original_sha256']
                if source_changed and not allow_source_update:
                    raise BrandingError('Native source is missing or changed: ' + relative)
                if source_changed and (native_hash is None or native_hash != expected_source_hash):
                    raise BrandingError('New native source does not match the integration snapshot: ' + relative)
                if source_changed and record['state'] != 'published':
                    raise BrandingError('Resume the previous branding transaction before updating its source: ' + relative)
                if not source_changed and expected_source_hash and expected_source_hash != record['original_sha256']:
                    raise BrandingError('Native snapshot differs from branding provenance: ' + relative)
                if not source_changed and not policy_changed:
                    if before_hash == record['published_sha256']:
                        if guard:
                            guard()
                        self.policy.assert_current()
                        if identity(public) != before_version:
                            raise BrandingError('Published image changed while verifying branding: ' + relative)
                        if identity(record_path) != record_version:
                            raise BrandingError('Branding record changed during verification: ' + relative)
                        if record['state'] != 'published':
                            record = self.save_record(relative, {**record, 'state': 'published'})
                        return {'status': 'skipped', **self.proof(record)}
                    source_bytes = (self.state / record['backup']).read_bytes()
                else:
                    atomic_json(self.state / 'history' / self.key(relative) / (record['record_sha256'] + '.json'),
                                record, immutable=True)
                    source_bytes = native.read_bytes() if source_changed else (self.state / record['backup']).read_bytes()
                    record = None
            else:
                if native is not None:
                    if not native.is_file():
                        raise BrandingError('Native source is missing: ' + relative)
                    source_bytes = native.read_bytes()
                    native_hash = hashlib.sha256(source_bytes).hexdigest()
                    allowed = {native_hash}
                    if allow_source_update and expected_existing_hash:
                        allowed.add(expected_existing_hash)
                    if before_hash is not None and before_hash not in allowed:
                        raise BrandingError('Unrecorded public image differs from its native source: ' + relative)
                else:
                    if not public.is_file():
                        raise BrandingError('Original teaching image is missing: ' + relative)
                    source_bytes = public.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            if expected_source_hash and expected_source_hash != source_hash:
                raise BrandingError('Native snapshot changed before branding: ' + relative)
            payload, details = render(source_bytes, self.policy)
            published_hash = hashlib.sha256(payload).hexdigest()
            if record and (record['published_sha256'] != published_hash
                           or any(record.get(key) != value for key, value in details.items())):
                raise BrandingError('Prepared branding transaction is not reproducible: ' + relative)
            backup = self.state / 'originals' / source_hash
            atomic_bytes(backup, source_bytes, immutable=True)
            if record is None:
                record = {
                    'version': 1, 'state': 'prepared', 'path': relative,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'original_sha256': source_hash, 'published_sha256': published_hash,
                    'original_bytes': len(source_bytes), 'published_bytes': len(payload),
                    'expected_public_sha256': before_hash, 'backup': 'originals/' + source_hash,
                    'native_source': relative_path(self.root, native) if native is not None else None,
                    'policy_sha256': self.policy.fingerprint, 'policy': self.policy.provenance,
                    **details,
                }

            def assert_unchanged():
                self.policy.assert_current()
                if native is not None and file_hash(native) != source_hash:
                    raise BrandingError('Native source changed while preparing branding: ' + relative)
                if guard:
                    guard()
                if (identity(public) if public.exists() else None) != before_version:
                    raise BrandingError('Published image changed while preparing branding: ' + relative)
                if (identity(record_path) if record_path.exists() else None) != record_version:
                    raise BrandingError('Branding record changed while preparing publication: ' + relative)

            assert_unchanged()
            record = self.save_record(relative, record)
            record_version = identity(record_path)
            atomic_bytes(public, payload, guard=assert_unchanged)
            record = self.save_record(relative, {**record, 'state': 'published'})
            return {'status': 'applied', **self.proof(record), 'method': record['method'],
                    'published_dimensions': record['published_dimensions']}

    def sample(self, relative, destination):
        row = self.inspect(relative)
        source = self.state / row['backup'] if 'backup' in row else self.path(relative)
        payload, details = render(source.read_bytes(), self.policy)
        target = Path(destination) / relative
        # Preview commands may never write into the live public tree.
        if target.resolve().is_relative_to(self.root / 'src/.vuepress/public'):
            raise BrandingError('Sample destination must be outside public')
        atomic_bytes(target, payload)
        return {'path': relative, 'sample': str(target), 'status': 'sampled',
                'original_sha256': row['original_sha256'],
                'published_sha256': hashlib.sha256(payload).hexdigest(), **details}


def verify_home_previews(root, policy=None):
    """Verify each preview's actual bytes and its current (already branded) source."""
    root = Path(root).resolve()
    public = root / 'src/.vuepress/public'
    manifest = read_json(public / 'images/home-previews/manifest.json')
    store = BrandingStore(root, policy) if policy else None
    if policy and manifest.get('brandingPolicySha256') != policy.fingerprint:
        raise BrandingError('Homepage preview branding policy is missing or stale')
    rows = manifest.get('images')
    if not isinstance(rows, list) or not rows:
        raise BrandingError('Homepage preview inventory is missing')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BrandingError('Invalid homepage preview record')
        source_url, preview_url = row.get('source', ''), row.get('preview', '')
        if (not source_url.startswith('/images/content-rebuild/')
                or not preview_url.startswith('/images/home-previews/') or preview_url in seen):
            raise BrandingError('Invalid or duplicate homepage preview path')
        seen.add(preview_url)
        source = rooted(public, source_url.lstrip('/'))
        preview = rooted(public, preview_url.lstrip('/'))
        for path, prefix in ((source, 'source'), (preview, 'preview')):
            if (not path.is_file() or file_hash(path) != row.get(prefix + 'Sha256')
                    or path.stat().st_size != row.get(prefix + 'Bytes')):
                raise BrandingError('Homepage preview source or output changed: ' + str(path))
            with Image.open(path) as image:
                if list(image.size) != row.get(prefix + 'Dimensions'):
                    raise BrandingError('Homepage preview dimensions differ from its record: ' + str(path))
        if store:
            record = store.inspect(source.relative_to(store.public).as_posix(), require_published=True)
            if row.get('sourceBranding') != store.proof(record):
                raise BrandingError('Homepage preview source branding is stale: ' + preview_url)
    return {'passed': True, 'images': len(rows)}
