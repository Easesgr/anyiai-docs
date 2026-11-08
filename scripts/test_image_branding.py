"""Publication provenance, interruption recovery, and pixel preservation regressions."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

import image_branding as BRANDING


def script(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REBUILD = script('branding_rebuild', 'rebuild-content.py')
VALIDATE = script('branding_validate', 'validate-rebuilt-content.py')
PREVIEWS = script('branding_previews', 'prepare-home-previews.py')
CLI = script('branding_cli', 'brand-content-images.py')


class BrandingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / 'src/.vuepress/public/images'
        self.public.mkdir(parents=True)
        self.out = self.root / 'output/content-rebuild'
        self.logo = self.public / 'branding/logo.png'
        self.logo.parent.mkdir()
        logo = Image.new('RGBA', (120, 60))
        draw = ImageDraw.Draw(logo)
        draw.rectangle((10, 10, 60, 49), fill=(255, 0, 80, 255))
        draw.rectangle((65, 10, 109, 49), fill=(0, 160, 100, 255))
        logo.save(self.logo)
        self.config = self.root / 'config/image-branding.json'
        self.config.parent.mkdir()
        self.config.write_text(json.dumps({
            'version': 1, 'enabled': True, 'logo': self.logo.relative_to(self.root).as_posix(),
            'logo_version': 'test-exact-raster-v1',
            'exclude': ['branding/**', 'home-previews/**', 'wechat-qrcode.png'],
        }))
        self.policy = BRANDING.load_policy(self.root)
        self.store = BRANDING.BrandingStore(self.root, self.policy)

    def add_image(self, relative='legacy/lesson.png', color=(250, 249, 246), native=False):
        target = self.public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (400, 240), color)
        ImageDraw.Draw(image).rectangle((20, 20, 100, 90), fill='navy')
        image.save(target, lossless=True)
        if native:
            source = self.store.native_for(relative)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(target.read_bytes())
        return target

    def test_absent_policy_is_opt_in(self):
        self.config.unlink()
        self.assertIsNone(BRANDING.load_policy(self.root))

    def test_overlay_changes_only_logo_rectangle(self):
        public = self.add_image()
        original = public.read_bytes()
        result = self.store.publish('legacy/lesson.png')
        self.assertEqual(result['method'], 'bottom-right-overlay')
        record = self.store.inspect('legacy/lesson.png', deep=True, require_published=True)
        self.assertEqual(record['original_dimensions'], record['published_dimensions'])
        self.assertEqual((self.store.state / record['backup']).read_bytes(), original)
        with Image.open(io.BytesIO(original)) as source, Image.open(public) as published:
            changed = ImageChops.difference(source, published).getbbox()
        x1, y1, x2, y2 = record['logo_bbox']
        self.assertGreaterEqual(changed[0], x1)
        self.assertGreaterEqual(changed[1], y1)
        self.assertLessEqual(changed[2], x2)
        self.assertLessEqual(changed[3], y2)

    def test_busy_corner_overlay_keeps_original_canvas_and_surrounding_pixels(self):
        for suffix in ['png', 'webp']:
            with self.subTest(format=suffix):
                relative = 'legacy/busy.' + suffix
                public = self.add_image(relative, color=(20, 50, 100))
                with Image.open(public) as image:
                    original_pixels = image.convert('RGB').copy()
                self.store.publish(relative)
                record = self.store.inspect(relative, deep=True)
                self.assertEqual(record['method'], 'bottom-right-overlay')
                self.assertEqual(record['published_dimensions'], [400, 240])
                self.assertLess(record['logo_bbox'][1], 240)
                self.assertLessEqual(record['logo_bbox'][3], 240)
                with Image.open(public) as image:
                    published_pixels = image.convert('RGB')
                box = record['logo_bbox']
                published_pixels.paste(original_pixels.crop(box), box)
                self.assertIsNone(ImageChops.difference(original_pixels, published_pixels).getbbox())

    def legacy_footer(self, relative='legacy/lesson.png', native=False):
        public = self.add_image(relative, color=(20, 50, 100), native=native)
        original = public.read_bytes()
        self.store.publish(relative)
        record = self.store.load_record(relative)
        with Image.open(io.BytesIO(original)) as source:
            footer = Image.new('RGB', (source.width, source.height + 30), (250, 249, 246))
            footer.paste(source, (0, 0))
            footer.paste((255, 0, 80), (350, 245, 390, 265))
            footer.save(public)
        record['policy'] = json.loads(json.dumps(record['policy']))
        record['policy']['engine'] = 'anyi-image-branding-v1'
        record['policy'].pop('composition')
        record['policy']['placement'].update({'padding_fraction': 0.006, 'footer_rgb': [250, 249, 246]})
        record['policy']['quiet_rule'] = 'all RGB >=224, chroma <=18, channel range <=12, stddev <=4.5, alpha >=254'
        record.update({'policy_sha256': BRANDING.value_hash(record['policy']),
                       'published_sha256': BRANDING.file_hash(public), 'published_bytes': public.stat().st_size,
                       'published_dimensions': [400, 270], 'logo_bbox': [350, 245, 390, 265], 'method': 'footer'})
        self.store.save_record(relative, record)
        return public, original

    def test_policy_correction_removes_footer_using_original_backup(self):
        relative = 'content-rebuild/sample/cover.png'
        public, original = self.legacy_footer(relative, native=True)
        previous_record = self.store.record_path(relative).read_bytes()
        with self.assertRaisesRegex(BRANDING.BrandingError, 'Stale branding'):
            self.store.publish(relative)
        self.store.publish(relative, allow_policy_update=True)
        record = self.store.inspect(relative, deep=True, require_published=True)
        self.assertEqual(record['published_dimensions'], [400, 240])
        self.assertEqual(record['method'], 'bottom-right-overlay')
        self.assertEqual((self.store.state / record['backup']).read_bytes(), original)
        self.assertEqual(self.store.native_for(relative).read_bytes(), original)
        history = list((self.store.state / 'history').rglob('*.json'))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].read_bytes(), previous_record)
        published = public.read_bytes()
        self.assertEqual(self.store.publish(relative, allow_policy_update=True)['status'], 'skipped')
        self.assertEqual(public.read_bytes(), published)

    def test_policy_correction_preserves_external_public_edits(self):
        public, _ = self.legacy_footer()
        public.write_bytes(public.read_bytes() + b'user edit')
        edited = public.read_bytes()
        with self.assertRaisesRegex(BRANDING.BrandingError, 'outside the branding'):
            self.store.publish('legacy/lesson.png', allow_policy_update=True)
        self.assertEqual(public.read_bytes(), edited)

    def test_interrupted_policy_correction_resumes_without_footer_or_duplicate_logo(self):
        public, original = self.legacy_footer()
        before = public.read_bytes()
        writer = BRANDING.atomic_bytes

        def crash_on_public(path, *args, **kwargs):
            if path.resolve() == public.resolve():
                raise OSError('simulated correction interruption')
            return writer(path, *args, **kwargs)

        with patch.object(BRANDING, 'atomic_bytes', side_effect=crash_on_public):
            with self.assertRaisesRegex(OSError, 'interruption'):
                self.store.publish('legacy/lesson.png', allow_policy_update=True)
        self.assertEqual(public.read_bytes(), before)
        self.store.publish('legacy/lesson.png')
        expected, _ = BRANDING.render(original, self.policy)
        self.assertEqual(public.read_bytes(), expected)
        self.store.inspect('legacy/lesson.png', deep=True, require_published=True)

    def test_native_and_receipt_are_unchanged_and_second_run_is_a_noop(self):
        relative = 'content-rebuild/sample/cover.png'
        public = self.add_image(relative, native=True)
        source = self.store.native_for(relative)
        receipt = source.with_suffix('.json')
        receipt.write_text('{"provider":"untouched"}')
        original, receipt_bytes = source.read_bytes(), receipt.read_bytes()
        self.store.publish(relative)
        published, version = public.read_bytes(), BRANDING.identity(public)
        record = self.store.record_path(relative).read_bytes()
        self.assertEqual(self.store.publish(relative)['status'], 'skipped')
        self.assertEqual(public.read_bytes(), published)
        self.assertEqual(BRANDING.identity(public), version)
        self.assertEqual(self.store.record_path(relative).read_bytes(), record)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(receipt.read_bytes(), receipt_bytes)
        self.assertNotEqual(original, published)

    def test_changed_public_is_never_overwritten(self):
        public = self.add_image()
        self.store.publish('legacy/lesson.png')
        public.write_bytes(public.read_bytes() + b'edited by user')
        changed = public.read_bytes()
        for action in (self.store.inspect, self.store.publish):
            with self.assertRaisesRegex(BRANDING.BrandingError, 'outside the branding'):
                action('legacy/lesson.png')
        self.assertEqual(public.read_bytes(), changed)

    def test_changed_native_requires_explicit_integration_snapshot(self):
        relative = 'content-rebuild/sample/cover.png'
        public = self.add_image(relative, native=True)
        self.store.publish(relative)
        source = self.store.native_for(relative)
        Image.new('RGB', (400, 240), 'blue').save(source)
        published = public.read_bytes()
        for action in (self.store.inspect, self.store.publish):
            with self.assertRaisesRegex(BRANDING.BrandingError, 'Native source'):
                action(relative)
        self.assertEqual(public.read_bytes(), published)
        self.store.publish(relative, source=source, expected_source_hash=BRANDING.file_hash(source), allow_source_update=True)
        self.store.inspect(relative, deep=True, require_published=True)
        self.assertEqual(len(list((self.store.state / 'history').rglob('*.json'))), 1)

    def test_logo_and_policy_changes_invalidate_existing_records(self):
        self.add_image()
        self.store.publish('legacy/lesson.png')
        original_logo = self.logo.read_bytes()
        self.logo.write_bytes(original_logo + b'logo revision')
        with self.assertRaisesRegex(BRANDING.BrandingError, 'logo changed'):
            self.store.inspect('legacy/lesson.png')
        with self.assertRaisesRegex(BRANDING.BrandingError, 'Stale branding'):
            BRANDING.BrandingStore(self.root, BRANDING.load_policy(self.root)).inspect('legacy/lesson.png')
        self.logo.write_bytes(original_logo)
        config = json.loads(self.config.read_text())
        config['placement'] = {'width_fraction': 0.15}
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(BRANDING.BrandingError, 'Stale branding'):
            BRANDING.BrandingStore(self.root, BRANDING.load_policy(self.root)).publish('legacy/lesson.png')

    def test_stale_provenance_and_missing_original_are_detected(self):
        self.add_image()
        self.store.publish('legacy/lesson.png')
        record_path = self.store.record_path('legacy/lesson.png')
        original = record_path.read_bytes()
        record = json.loads(original)
        record['published_sha256'] = '0' * 64
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(BRANDING.BrandingError, 'Invalid branding provenance'):
            self.store.inspect('legacy/lesson.png')
        record_path.write_bytes(original)
        (self.store.state / json.loads(original)['backup']).unlink()
        with self.assertRaisesRegex(BRANDING.BrandingError, 'backup is missing or changed'):
            self.store.inspect('legacy/lesson.png')

    def test_concurrent_public_edit_during_render_is_preserved(self):
        public = self.add_image()
        renderer = BRANDING.render

        def edit_during_render(*args):
            result = renderer(*args)
            public.write_bytes(b'user edit while rendering')
            return result

        with patch.object(BRANDING, 'render', side_effect=edit_during_render):
            with self.assertRaisesRegex(BRANDING.BrandingError, 'changed while preparing'):
                self.store.publish('legacy/lesson.png')
        self.assertEqual(public.read_bytes(), b'user edit while rendering')

    def test_concurrent_publishers_share_one_committed_derivative(self):
        self.add_image()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.store.publish, 'legacy/lesson.png') for _ in range(2)]
            results = [future.result() for future in futures]
        self.assertEqual(sorted(row['status'] for row in results), ['applied', 'skipped'])
        self.assertEqual(results[0]['published_sha256'], results[1]['published_sha256'])
        self.store.inspect('legacy/lesson.png', deep=True, require_published=True)

    def test_native_change_while_rendering_does_not_publish_stale_bytes(self):
        relative = 'content-rebuild/sample/cover.png'
        public = self.add_image(relative, native=True)
        original = public.read_bytes()
        source = self.store.native_for(relative)
        renderer = BRANDING.render

        def edit_native(*args):
            result = renderer(*args)
            Image.new('RGB', (400, 240), 'red').save(source)
            return result

        with patch.object(BRANDING, 'render', side_effect=edit_native):
            with self.assertRaisesRegex(BRANDING.BrandingError, 'Native source changed while preparing'):
                self.store.publish(relative)
        self.assertEqual(public.read_bytes(), original)

    def test_public_edit_after_durable_prepare_is_preserved(self):
        public = self.add_image()
        save = self.store.save_record

        def edit_after_prepare(relative, record):
            result = save(relative, record)
            public.write_bytes(b'edited after journal write')
            return result

        with patch.object(self.store, 'save_record', side_effect=edit_after_prepare):
            with self.assertRaisesRegex(BRANDING.BrandingError, 'changed while preparing'):
                self.store.publish('legacy/lesson.png')
        self.assertEqual(public.read_bytes(), b'edited after journal write')

    def test_unrecorded_public_differences_are_not_treated_as_native(self):
        relative = 'content-rebuild/sample/cover.png'
        public = self.add_image(relative, native=True)
        public.write_bytes(public.read_bytes() + b'user edit before first apply')
        edited = public.read_bytes()
        with self.assertRaisesRegex(BRANDING.BrandingError, 'differs from its native source'):
            self.store.publish(relative)
        self.assertEqual(public.read_bytes(), edited)

    def test_interruption_before_public_replace_resumes_from_original(self):
        public = self.add_image()
        original = public.read_bytes()
        writer = BRANDING.atomic_bytes

        def crash_on_public(path, *args, **kwargs):
            if path.resolve() == public.resolve():
                raise OSError('simulated process interruption')
            return writer(path, *args, **kwargs)

        with patch.object(BRANDING, 'atomic_bytes', side_effect=crash_on_public):
            with self.assertRaisesRegex(OSError, 'interruption'):
                self.store.publish('legacy/lesson.png')
        self.assertEqual(public.read_bytes(), original)
        self.assertEqual(self.store.inspect('legacy/lesson.png')['status'], 'prepared')
        self.store.publish('legacy/lesson.png')
        self.store.inspect('legacy/lesson.png', deep=True, require_published=True)

    def test_interruption_after_replace_finishes_record_without_stamping_twice(self):
        public = self.add_image()
        save = self.store.save_record

        def crash_on_commit(relative, record):
            if record['state'] == 'published':
                raise OSError('simulated commit interruption')
            return save(relative, record)

        with patch.object(self.store, 'save_record', side_effect=crash_on_commit):
            with self.assertRaisesRegex(OSError, 'interruption'):
                self.store.publish('legacy/lesson.png')
        published = public.read_bytes()
        self.assertEqual(self.store.inspect('legacy/lesson.png')['status'], 'recoverable')
        self.assertEqual(self.store.publish('legacy/lesson.png')['status'], 'skipped')
        self.assertEqual(public.read_bytes(), published)
        self.store.inspect('legacy/lesson.png', deep=True, require_published=True)

    def test_dry_run_sample_exclusions_and_missing_file_verification(self):
        public = self.add_image('root-teaching.png')
        self.add_image('wechat-qrcode.png')
        self.add_image('home-previews/derived.webp')
        original = public.read_bytes()
        report = self.root / 'dry-run.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(['dry-run', '--root', str(self.root), '--report', str(report)]), 0)
            self.assertEqual(CLI.main(['sample', '--root', str(self.root), '--only', 'root-teaching.png',
                                       '--destination', str(self.root / 'samples')]), 0)
        self.assertEqual(json.loads(report.read_text())['total'], 1)
        self.assertEqual(public.read_bytes(), original)
        self.assertFalse(self.store.state.exists())
        self.store.publish('root-teaching.png')
        public.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(['verify', '--root', str(self.root)]), 1)

    def prepare_page(self, branding=True):
        if not branding:
            self.config.unlink()
        self.out.mkdir(parents=True, exist_ok=True)
        self.src = self.root / 'src'
        patcher = patch.multiple(REBUILD, ROOT=self.root, SRC=self.src, OUT=self.out)
        patcher.start()
        self.addCleanup(patcher.stop)
        validate_patcher = patch.object(VALIDATE, 'REBUILD', REBUILD)
        validate_patcher.start()
        self.addCleanup(validate_patcher.stop)
        self.page = {'id': 'sample', 'path': 'topic.md', 'title': 'Original', 'frontmatter': {'title': 'Original'}}
        self.folder = REBUILD.page_folder(self.page)
        self.folder.mkdir(parents=True)
        original = '---\ntitle: Original\n---\n\nOriginal content\n'
        (self.folder / 'original.md').write_text(original)
        (self.src / 'topic.md').write_text(original)
        self.page['source_sha256'] = REBUILD.digest(original)
        self.article = {'title': 'Updated', 'description': 'Description',
                        'body_markdown': '# Updated\n\n![Relationship](rebuild:cover)\n\nExplanation.',
                        'images': [{'id': 'cover', 'alt': 'Relationship', 'prompt': 'Exact diagram brief'}]}
        REBUILD.write_json(self.folder / 'reviewed.json', self.article)
        REBUILD.write_json(self.folder / 'editorial-polish.json', {
            'body_sha256': REBUILD.digest(self.article['body_markdown']), 'title': self.article['title'], 'diagram_issues': [],
        })
        native = self.folder / 'images/cover.png'
        native.parent.mkdir()
        Image.new('RGB', (400, 240), (250, 249, 246)).save(native)
        REBUILD.write_json(native.with_suffix('.json'), {
            'request': {'prompt': '用途：Updated\n本图说明：Relationship\n具体画面：Exact diagram brief\n统一使用视觉智构安逸i的「马克笔」风格。'},
            'receipt': {'status': 'succeeded', 'path': str(native)}, 'asset_sha256': REBUILD.file_digest(native),
        })
        REBUILD.write_json(self.out / 'publication-policy.json', {'visual_review': 'skipped_by_user'})
        REBUILD.write_json(self.out / 'inventory.json', {'pages': [self.page]})
        REBUILD.write_json(self.out / 'navigation.json', {'link': '/topic.md'})
        REBUILD.write_json(self.folder / 'final-audit.json', {'body_sha256': REBUILD.digest(self.article['body_markdown'])})
        component = self.src / '.vuepress/components/HomePortal.vue'
        component.parent.mkdir(parents=True)
        component.write_text('<template>Teaching portal</template>')
        return native

    def test_no_policy_preserves_exact_native_integration_and_strict_validation(self):
        native = self.prepare_page(branding=False)
        REBUILD.integrate([self.page])
        public = self.public / 'content-rebuild/sample/cover.png'
        self.assertEqual(public.read_bytes(), native.read_bytes())
        self.assertNotIn('published_images', REBUILD.read_json(self.folder / 'integrated.json'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(VALIDATE.validate()['passed'])
        public.write_bytes(public.read_bytes() + b'changed')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(VALIDATE.validate()['passed'])

    def test_integration_and_validator_preserve_native_snapshot_and_branding(self):
        native = self.prepare_page()
        source_hash = REBUILD.file_digest(native)
        snapshot = REBUILD.media_snapshot(self.page)
        REBUILD.integrate([self.page])
        public = self.public / 'content-rebuild/sample/cover.png'
        first = public.read_bytes()
        self.assertEqual(REBUILD.media_snapshot(self.page), snapshot)
        integrated = REBUILD.read_json(self.folder / 'integrated.json')
        self.assertEqual(integrated['media_snapshot'], snapshot)
        self.assertEqual(integrated['published_images']['cover']['original_sha256'], source_hash)
        self.assertEqual(integrated['visual_review'], 'skipped_by_user')
        self.assertFalse((self.folder / 'visual-review.json').exists())
        REBUILD.integrate([self.page])
        self.assertEqual(public.read_bytes(), first)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(VALIDATE.validate()['passed'])
            self.assertEqual(REBUILD.status([self.page])['integrated_current'], 1)
        public.write_bytes(public.read_bytes() + b'changed')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(VALIDATE.validate()['passed'])
            self.assertEqual(REBUILD.status([self.page])['integrated_current'], 0)

    def test_bulk_branding_of_existing_integration_validates_without_rewriting_native_record(self):
        self.prepare_page(branding=False)
        REBUILD.integrate([self.page])
        integrated = (self.folder / 'integrated.json').read_bytes()
        self.config.write_text(json.dumps({'version': 1, 'enabled': True,
                                          'logo': self.logo.relative_to(self.root).as_posix(), 'logo_version': 'test'}))
        store = BRANDING.BrandingStore(self.root, BRANDING.load_policy(self.root))
        store.publish('content-rebuild/sample/cover.png')
        self.assertEqual((self.folder / 'integrated.json').read_bytes(), integrated)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(VALIDATE.validate()['passed'])

    def test_source_edit_during_branding_blocks_article_and_public_replacement(self):
        self.prepare_page()
        renderer = BRANDING.render

        def change_source(*args):
            result = renderer(*args)
            (self.src / 'topic.md').write_text('New user draft')
            return result

        with patch.object(BRANDING, 'render', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'Source changed during integration'):
                REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), 'New user draft')
        self.assertFalse((self.public / 'content-rebuild/sample/cover.png').exists())

    def test_integration_accepts_new_receipted_native_and_updates_public_proof(self):
        native = self.prepare_page()
        REBUILD.integrate([self.page])
        previous = REBUILD.read_json(self.folder / 'integrated.json')['published_images']['cover']
        Image.new('RGB', (400, 240), 'navy').save(native)
        receipt = REBUILD.read_json(native.with_suffix('.json'))
        receipt['asset_sha256'] = REBUILD.file_digest(native)
        REBUILD.write_json(native.with_suffix('.json'), receipt)
        REBUILD.integrate([self.page])
        current = REBUILD.read_json(self.folder / 'integrated.json')['published_images']['cover']
        self.assertNotEqual(previous['original_sha256'], current['original_sha256'])
        self.assertEqual(current['original_sha256'], REBUILD.file_digest(native))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(VALIDATE.validate()['passed'])

    def test_previews_inherit_branding_once_and_manifest_matches_actual_bytes(self):
        cover_id = '0123456789abcdef'
        relative = f'content-rebuild/{cover_id}/cover.png'
        self.add_image(relative, color='navy', native=True)
        self.store.publish(relative)
        component = self.root / 'HomePortal.vue'
        component.write_text("const cover = '" + cover_id + "';")
        stages = self.root / 'learning.ts'
        stages.write_text("const cover = '/images/" + relative + "';")
        destination = self.public / 'home-previews'
        with patch.multiple(PREVIEWS, ROOT=self.root, PUBLIC=self.public.parent, COMPONENT=component,
                            STAGES=stages, DESTINATION=destination), contextlib.redirect_stdout(io.StringIO()):
            PREVIEWS.main()
            target = destination / (cover_id + '.webp')
            first, first_version = target.read_bytes(), BRANDING.identity(target)
            self.assertEqual(BRANDING.verify_home_previews(self.root, self.policy)['images'], 1)
            PREVIEWS.main()
            self.assertEqual(target.read_bytes(), first)
            self.assertEqual(BRANDING.identity(target), first_version)
            target.write_bytes(first + b'user edit')
            with self.assertRaisesRegex(BRANDING.BrandingError, 'source or output changed'):
                BRANDING.verify_home_previews(self.root, self.policy)
            with self.assertRaisesRegex(BRANDING.BrandingError, 'outside its manifest'):
                PREVIEWS.main()


if __name__ == '__main__':
    unittest.main()
