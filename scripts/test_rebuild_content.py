"""Pipeline version/repair regressions, with no model or image provider calls."""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

SPEC = importlib.util.spec_from_file_location('rebuild_content', Path(__file__).with_name('rebuild-content.py'))
REBUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REBUILD)


class NavigationLabelTests(unittest.TestCase):
    def test_existing_short_labels_survive_nested_sidebar_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            sidebar = Path(folder) / 'sidebar.ts'
            sidebar.write_text('import { sidebar } from "vuepress-theme-hope";\n'
                               'export const zhSidebar = sidebar('
                               + json.dumps({'/learning-path/': [{
                                   'text': '学习路线', 'link': '/learning-path/',
                                   'children': [{'text': 'RAG 实战', 'link': '/learning-path/rag-practice.md'}],
                               }]}) + ');\n')
            self.assertEqual(REBUILD.read_sidebar_labels(sidebar), {
                '/learning-path/': '学习路线', '/learning-path/rag-practice.md': 'RAG 实战',
            })
            self.assertEqual(REBUILD.read_sidebar_labels(Path(folder) / 'missing.ts'), {})


class FakeImageRequest:
    def __init__(self, page, article, image):
        self.id = page['id'] + '-' + image['id']
        self.prompt = (f"用途：{article['title']}\n本图说明：{image['alt']}\n具体画面：{image['prompt']}"
                       '\n统一使用视觉智构安逸i的「马克笔」风格。\n准确标签与人类讲解员。')

    def model_dump(self):
        return {'id': self.id, 'prompt': self.prompt, 'aspect_ratio': '16:9'}


class ContentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.src = self.root / 'src'
        self.out = self.root / 'output'
        self.src.mkdir()
        self.out.mkdir()
        self.overrides = patch.multiple(REBUILD, ROOT=self.root, SRC=self.src, OUT=self.out)
        self.overrides.start()
        self.addCleanup(self.overrides.stop)
        self.events = patch.object(REBUILD, 'event')
        self.events.start()
        self.addCleanup(self.events.stop)
        self.original = '---\ntitle: 原标题\nicon: book\n---\n\n# 原标题\n原文\n'
        self.page = {
            'id': 'sample-page', 'path': 'topic.md', 'title': '原标题', 'kind': 'article', 'section': 'fundamentals',
            'source_sha256': REBUILD.digest(self.original), 'frontmatter': {'title': '原标题', 'icon': 'book'},
        }
        (self.src / 'topic.md').write_text(self.original)
        self.folder = REBUILD.page_folder(self.page)
        self.folder.mkdir(parents=True)
        (self.folder / 'original.md').write_text(self.original)
        self.article = {
            'title': '新标题', 'description': '摘要',
            'body_markdown': '# 新标题\n\n![具体关系](rebuild:cover)\n\n新的解释。',
            'images': [{'id': 'cover', 'alt': '具体关系', 'prompt': '明确画面与人类参与、正确箭头关系。' * 15}],
        }
        REBUILD.write_json(self.folder / 'reviewed.json', self.article)

    def add_polish(self):
        article = REBUILD.read_json(self.folder / 'reviewed.json')
        REBUILD.write_json(self.folder / 'editorial-polish.json', {
            'body_sha256': REBUILD.digest(article['body_markdown']), 'title': article['title'], 'diagram_issues': [],
        })

    def valid_verdict(self, passed=True):
        return {'passed': passed, 'images': [{'id': 'cover', 'human_present': passed,
                                             'style_matches': True, 'explanation_matches': True,
                                             'issues': [] if passed else ['未看到参与讲解的人类角色']}]}

    def add_review(self, passed=True):
        snapshot = REBUILD.media_snapshot(self.page)
        self.assertIsNotNone(snapshot)
        REBUILD.write_json(self.folder / 'visual-review.json', {**self.valid_verdict(passed), **snapshot})

    def save_image(self, payload=b'original-provider-image-bytes', native=False):
        provider = self.folder / ('provider-' + str(len(list(self.folder.glob('provider-*')))) + '.png')
        if native:
            Image.new('RGB', (160, 90), 'blue').save(provider)
        else:
            provider.write_bytes(payload)
        article = REBUILD.read_json(self.folder / 'reviewed.json')
        request = FakeImageRequest(self.page, article, article['images'][0])
        receipt = {'id': request.id, 'path': str(provider), 'status': 'succeeded'}
        target = self.folder / 'images/cover.png'
        REBUILD.publish_image(target, request, receipt)
        return target

    def add_image(self, passed=True, native=False):
        self.add_polish()
        target = self.save_image(native=native)
        self.add_review(passed)
        return target

    def change_body(self):
        article = REBUILD.read_json(self.folder / 'reviewed.json')
        article['body_markdown'] += '\n最终修订的事实。'
        REBUILD.write_json(self.folder / 'reviewed.json', article)
        self.add_polish()

    def adjudication(self, decision='false_positive'):
        review = REBUILD.read_json(self.folder / 'visual-review.json')
        concerns = REBUILD.repair_concerns(review['images'][0])
        return {
            'id': 'cover', 'decision': decision, 'human_present': decision == 'false_positive',
            'style_matches': True, 'explanation_matches': True,
            'issues': [] if decision == 'false_positive' else ['原图左侧缺少人类讲解员'],
            'native_evidence': [{'concern_id': concern['id'], 'location': '画面左侧讲解区',
                                 'observed': '原图中可见完整脸部、身体及指向流程的手臂' if decision == 'false_positive'
                                 else '左侧只有机器人图标，没有人类角色',
                                 'reason': '对照原始尺寸图片和正文中讲解员操作的图解任务逐项核对',
                                 'assessment': {'false_positive': 'false_positive', 'redraw': 'confirmed',
                                                'unresolved': 'uncertain'}[decision]}
                                for concern in concerns],
            'corrected_prompt': '浅灰底、彩色马克笔粗黑线，完整成年黑短发蓝衣黄内搭人类操作图中节点，准确标注箭头。' * 8,
        }

    def assert_pending(self):
        REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), self.original)
        self.assertFalse((self.folder / 'integrated.json').exists())
        self.assertEqual(json.loads((self.out / 'integration.json').read_text())['pending'], ['topic.md'])

    def test_missing_images_leave_original_untouched(self):
        self.assert_pending()

    def test_editorial_image_markers_accept_the_markdown_prefix(self):
        self.add_polish()
        editorial = REBUILD.read_json(self.folder / 'editorial-polish.json')
        editorial['diagram_issues'] = [{'id': 'rebuild:cover', 'problem': '增加具体协议边界标签'}]
        REBUILD.write_json(self.folder / 'editorial-polish.json', editorial)
        corrected = REBUILD.apply_diagram_corrections(self.page, self.article)
        self.assertIn('增加具体协议边界标签', corrected['images'][0]['prompt'])
        self.assertTrue(REBUILD.diagram_corrections_current(self.page))

    def test_unknown_editorial_image_ids_still_block_prompt_changes(self):
        self.add_polish()
        editorial = REBUILD.read_json(self.folder / 'editorial-polish.json')
        editorial['diagram_issues'] = [{'id': 'rebuild:fig-unknown', 'problem': '未知图'}]
        REBUILD.write_json(self.folder / 'editorial-polish.json', editorial)
        with self.assertRaisesRegex(ValueError, 'Unknown illustration'):
            REBUILD.apply_diagram_corrections(self.page, self.article)
        self.assertFalse((self.folder / 'diagram-corrections-applied.json').exists())

    def test_editorial_prose_can_change_while_existing_image_descriptions_are_restored(self):
        response = {**self.article, 'body_markdown': '# 新标题\n\n![新的说法](rebuild:cover)\n\n更清楚的解释。'}
        candidate = REBUILD.editorial_candidate(self.article, response)
        self.assertIn('更清楚的解释。', candidate['body_markdown'])
        self.assertEqual(REBUILD.IMAGE_MARKER.findall(candidate['body_markdown']),
                         REBUILD.IMAGE_MARKER.findall(self.article['body_markdown']))

    def test_editorial_anchor_changes_are_not_repaired_by_guessing(self):
        for body in ['# 标题\n无图', '# 标题\n![未知图](rebuild:missing)',
                     '# 标题\n![重复](rebuild:cover)\n![重复](rebuild:cover)']:
            response = {**self.article, 'body_markdown': body}
            self.assertEqual(REBUILD.editorial_candidate(self.article, response)['body_markdown'], body)

    def skip_visual_review(self):
        REBUILD.write_json(self.out / 'publication-policy.json', {'visual_review': 'skipped_by_user'})

    def test_user_can_integrate_native_images_without_a_visual_verdict(self):
        self.skip_visual_review()
        self.add_polish()
        self.save_image()
        REBUILD.integrate([self.page])
        self.assertEqual(REBUILD.read_json(self.folder / 'integrated.json')['visual_review'], 'skipped_by_user')
        self.assertFalse((self.folder / 'visual-review.json').exists())
        self.assertFalse(REBUILD.media_passed(self.page))
        self.assertTrue(REBUILD.media_ready(self.page))
        self.assertEqual(REBUILD.status([self.page])['integrated_current'], 1)

    def test_skipping_visual_review_still_requires_matching_native_receipts(self):
        self.skip_visual_review()
        self.add_polish()
        target = self.save_image()
        target.write_bytes(b'replaced-without-generation-receipt')
        self.assert_pending()

    def test_user_skip_keeps_media_watcher_on_generation_only(self):
        self.skip_visual_review()
        with patch.object(REBUILD, 'images', new=AsyncMock()) as generate, \
                patch.object(REBUILD, 'image_qa', new=AsyncMock()) as review, \
                patch.object(REBUILD, 'repair_images', new=AsyncMock()) as repair:
            asyncio.run(REBUILD.watch_media([self.page], 4))
        generate.assert_awaited_once_with([self.page], 4)
        review.assert_not_awaited()
        repair.assert_not_awaited()

    def test_skipping_visual_review_does_not_disable_source_edit_protection(self):
        self.skip_visual_review()
        self.add_polish()
        self.save_image()
        edited = self.original + '\n用户修改\n'
        (self.src / 'topic.md').write_text(edited)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), edited)

    def test_concurrent_source_edit_blocks_overwrite(self):
        self.add_image()
        edited = self.original + '\n用户刚编辑的新内容\n'
        (self.src / 'topic.md').write_text(edited)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), edited)

    def test_source_edit_while_staging_also_prevents_public_asset_overwrite(self):
        self.add_image()
        public = self.src / '.vuepress/public/images/content-rebuild/sample-page/cover.png'
        public.parent.mkdir(parents=True)
        public.write_bytes(b'previous-public-asset')
        copy = REBUILD.shutil.copy2
        edited = self.original + '\n用户同步写入的新段落\n'

        def edit_during_copy(source, destination):
            result = copy(source, destination)
            (self.src / 'topic.md').write_text(edited)
            return result

        with patch.object(REBUILD.shutil, 'copy2', side_effect=edit_during_copy):
            with self.assertRaisesRegex(ValueError, 'Source changed during integration'):
                REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), edited)
        self.assertEqual(public.read_bytes(), b'previous-public-asset')

    def test_complete_article_keeps_metadata_and_native_image_bytes(self):
        self.add_image()
        REBUILD.integrate([self.page])
        result = (self.src / 'topic.md').read_text()
        metadata, body = REBUILD.split_document(result)
        self.assertEqual(metadata['icon'], 'book')
        self.assertEqual(metadata['title'], '新标题')
        self.assertIn('/images/content-rebuild/sample-page/cover.png', body)
        self.assertNotIn('rebuild:', body)
        image = self.src / '.vuepress/public/images/content-rebuild/sample-page/cover.png'
        self.assertEqual(image.read_bytes(), b'original-provider-image-bytes')
        REBUILD.integrate([self.page])
        self.assertEqual((self.src / 'topic.md').read_text(), result)

    def test_integration_preserves_yaml_dates_after_json_inventory_round_trip(self):
        self.original = self.original.replace('icon: book\n', 'icon: book\ndate: 2026-07-16\n')
        (self.src / 'topic.md').write_text(self.original)
        (self.folder / 'original.md').write_text(self.original)
        self.page['source_sha256'] = REBUILD.digest(self.original)
        self.page['frontmatter']['date'] = '2026-07-16'
        self.add_image()

        REBUILD.integrate([self.page])

        result = (self.src / 'topic.md').read_text()
        metadata, _ = REBUILD.split_document(result)
        self.assertEqual(metadata['date'], date(2026, 7, 16))
        self.assertIn('date: 2026-07-16\n', result)
        self.assertEqual(REBUILD.read_json(self.folder / 'integrated.json')['content_sha256'],
                         REBUILD.digest(result))

    def test_duplicate_or_missing_image_anchors_reject_article(self):
        article = dict(self.article)
        article['body_markdown'] = '# 标题\n' + '有内容的正文。' * 200 + '\n![封面](rebuild:cover)\n![重复](rebuild:cover)'
        article['images'] = [{'id': name, 'alt': '关系', 'prompt': '明确关系和标签。' * 30}
                             for name in ['cover', 'fig-1', 'fig-2', 'fig-3', 'fig-4', 'fig-5', 'fig-6']]
        errors = REBUILD.validate_article(article, self.page)
        self.assertTrue(any('一一对应' in error for error in errors))

    def test_replacing_an_approved_image_requires_another_visual_review(self):
        self.add_image()
        self.save_image(b'changed-image-with-current-receipt')
        self.assert_pending()

    def test_unpolished_article_cannot_integrate_even_with_visual_pass(self):
        self.add_image()
        (self.folder / 'editorial-polish.json').unlink()
        self.assert_pending()

    def test_stale_polish_digest_blocks_integration(self):
        self.add_image()
        self.article['body_markdown'] += '\n未经主编审查的修改。'
        REBUILD.write_json(self.folder / 'reviewed.json', self.article)
        self.assert_pending()

    def test_stale_body_review_blocks_integration_even_when_polish_is_current(self):
        self.add_image()
        self.change_body()
        self.assertIsNotNone(REBUILD.polished_article(self.page))
        self.assert_pending()

    def test_changed_editorial_record_invalidates_review(self):
        self.add_image()
        editorial = REBUILD.read_json(self.folder / 'editorial-polish.json')
        editorial['corrections'] = ['新的主编核验说明']
        REBUILD.write_json(self.folder / 'editorial-polish.json', editorial)
        self.assert_pending()

    def test_legacy_review_without_complete_version_binding_requires_recheck(self):
        self.add_image()
        review = REBUILD.read_json(self.folder / 'visual-review.json')
        del review['image_receipt_hashes']
        REBUILD.write_json(self.folder / 'visual-review.json', review)
        self.assert_pending()

    def test_stale_prompt_receipt_blocks_existing_png_and_regenerates(self):
        old = self.add_image()
        old_bytes = old.read_bytes()
        self.article['images'][0]['prompt'] += '\n修正箭头方向并清楚标注边界。'
        REBUILD.write_json(self.folder / 'reviewed.json', self.article)
        self.assert_pending()
        generated = self.folder / 'fresh-provider.png'
        generated.write_bytes(b'corrected-native-image')

        async def generate(requests, concurrency, on_complete=None):
            self.assertEqual(concurrency, 1)
            self.assertIn('修正箭头方向', requests[0].prompt)
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}]}

        with patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate) as provider:
            asyncio.run(REBUILD.images([self.page], 1))
        provider.assert_awaited_once()
        self.assertEqual(old.read_bytes(), b'corrected-native-image')
        archived = list((self.folder / 'image-history/cover').glob('*/native.png'))
        self.assertEqual(archived[0].read_bytes(), old_bytes)
        self.assertIsNotNone(REBUILD.media_snapshot(self.page))
        self.assert_pending()

    def test_prompt_substring_is_not_a_matching_receipt(self):
        self.add_image()
        self.article['images'][0]['prompt'] = self.article['images'][0]['prompt'][:110]
        REBUILD.write_json(self.folder / 'reviewed.json', self.article)
        self.assertIsNone(REBUILD.current_image_record(self.folder, self.article['images'][0]))

    def test_receipt_must_belong_to_actual_native_asset(self):
        self.add_image()
        receipt_path = self.folder / 'images/cover.json'
        record = REBUILD.read_json(receipt_path)
        del record['asset_sha256']
        REBUILD.write_json(receipt_path, record)
        self.assertIsNotNone(REBUILD.current_image_record(self.folder, self.article['images'][0]))
        (self.folder / 'images/cover.png').write_bytes(b'unrelated-native-image')
        self.assertIsNone(REBUILD.current_image_record(self.folder, self.article['images'][0]))
        self.assert_pending()

    def test_top_level_pass_cannot_hide_failed_image_checks(self):
        self.add_image(False)
        review = REBUILD.read_json(self.folder / 'visual-review.json')
        review['passed'] = True
        REBUILD.write_json(self.folder / 'visual-review.json', review)
        self.assert_pending()

    def test_image_generation_discards_result_if_article_changes_during_await(self):
        self.add_polish()
        generated = self.folder / 'provider.png'
        generated.write_bytes(b'result-for-old-article')

        async def generate(requests, concurrency, on_complete=None):
            self.change_body()
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}]}

        with patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            asyncio.run(REBUILD.images([self.page], 1))
        self.assertFalse((self.folder / 'images/cover.png').exists())

    def test_streamed_image_is_not_overwritten_again_when_batch_returns(self):
        self.add_polish()
        generated = self.folder / 'provider.png'
        generated.write_bytes(b'first-generation')

        async def generate(requests, concurrency, on_complete=None):
            receipt = {'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}
            on_complete(requests[0], receipt)
            self.assertEqual((self.folder / 'images/cover.png').read_bytes(), b'first-generation')
            self.save_image(b'newer-asset-published-before-the-batch-finished')
            return {'results': [receipt]}

        with patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            asyncio.run(REBUILD.images([self.page], 1))
        self.assertEqual((self.folder / 'images/cover.png').read_bytes(),
                         b'newer-asset-published-before-the-batch-finished')

    def test_visual_qa_discards_verdict_if_article_changes_during_await(self):
        self.add_image(native=True)
        (self.folder / 'visual-review.json').unlink()

        async def ask(*args, **kwargs):
            self.change_body()
            return self.valid_verdict()

        with patch.object(REBUILD, 'ask', side_effect=ask):
            asyncio.run(REBUILD.image_qa([self.page], 1))
        self.assertFalse((self.folder / 'visual-review.json').exists())

    def test_late_contact_review_cannot_replace_newer_native_resolution(self):
        self.add_image(native=True)
        (self.folder / 'visual-review.json').unlink()

        async def inspect(*args, **kwargs):
            self.add_review(True)
            current = REBUILD.read_json(self.folder / 'visual-review.json')
            current['images'][0]['native_reinspection'] = {'record': 'newer-native-evidence.json'}
            REBUILD.write_json(self.folder / 'visual-review.json', current)
            return self.valid_verdict(False)

        with patch.object(REBUILD, 'ask', side_effect=inspect):
            asyncio.run(REBUILD.image_qa([self.page], 1))
        self.assertTrue(REBUILD.media_passed(self.page))
        self.assertIn('native_reinspection', REBUILD.current_visual_review(self.page)['images'][0])

    def test_stale_failed_review_does_not_trigger_native_repair(self):
        self.add_image(False)
        self.change_body()
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock) as ask:
            asyncio.run(REBUILD.repair_images([self.page], 1))
        ask.assert_not_called()
        self.assertFalse((self.folder / 'image-repairs').exists())

    def test_false_positive_requires_and_records_native_per_image_evidence(self):
        native = self.add_image(False)
        original_bytes = native.read_bytes()
        adjudication = self.adjudication()

        async def inspect(prompt, target, **kwargs):
            self.assertEqual(kwargs['image_paths'][0].read_bytes(), original_bytes)
            self.assertEqual(kwargs['image_paths'][0].name, 'native.png')
            self.assertIn(self.article['body_markdown'], prompt)
            return adjudication

        with patch.object(REBUILD, 'ask', side_effect=inspect), \
                patch.object(REBUILD, 'generate_image_requests', new_callable=AsyncMock) as provider:
            asyncio.run(REBUILD.repair_images([self.page], 1))
        provider.assert_not_called()
        self.assertTrue(REBUILD.media_passed(self.page))
        review = REBUILD.read_json(self.folder / 'visual-review.json')
        self.assertEqual(review['images'][0]['native_reinspection']['native_evidence'],
                         adjudication['native_evidence'])
        self.assertEqual(native.read_bytes(), original_bytes)

    def test_false_positive_without_native_evidence_stays_failed(self):
        self.add_image(False)
        adjudication = self.adjudication()
        adjudication['native_evidence'] = []
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication):
            remaining = asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=1))
        self.assertFalse(REBUILD.media_passed(self.page))
        self.assertTrue(remaining[0]['exhausted'])
        self.assert_pending()

    def test_false_positive_with_confirmed_or_uncertain_concern_is_rejected(self):
        self.add_image(False)
        adjudication = self.adjudication()
        adjudication['native_evidence'][0]['assessment'] = 'uncertain'
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication):
            asyncio.run(REBUILD.repair_images([self.page], 1))
        self.assertFalse(REBUILD.media_passed(self.page))

    def test_true_error_redraw_preserves_rejected_bytes_and_requires_new_qa(self):
        native = self.add_image(False, native=True)
        old_bytes = native.read_bytes()
        old_receipt = (self.folder / 'images/cover.json').read_bytes()
        replacement = self.folder / 'replacement-provider.png'
        Image.new('RGB', (160, 90), 'red').save(replacement)
        adjudication = self.adjudication('redraw')

        async def generate(requests, concurrency):
            self.assertIn(adjudication['corrected_prompt'], requests[0].prompt)
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(replacement)}]}

        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication), \
                patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            asyncio.run(REBUILD.repair_images([self.page], 1))
        history = next((self.folder / 'image-repairs/cover').iterdir())
        self.assertEqual((history / 'native.png').read_bytes(), old_bytes)
        self.assertEqual((history / 'native.json').read_bytes(), old_receipt)
        self.assertEqual(REBUILD.read_json(history / 'attempt.json')['status'], 'redrawn')
        self.assertEqual(native.read_bytes(), replacement.read_bytes())
        self.assertIsNotNone(REBUILD.media_snapshot(self.page))
        self.assert_pending()
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=self.valid_verdict()):
            asyncio.run(REBUILD.image_qa([self.page], 1))
        self.assertTrue(REBUILD.media_passed(self.page))
        REBUILD.integrate([self.page])
        published = self.src / '.vuepress/public/images/content-rebuild/sample-page/cover.png'
        self.assertEqual(published.read_bytes(), replacement.read_bytes())

    def test_repair_discards_adjudication_when_body_changes_during_await(self):
        native = self.add_image(False)
        old_bytes = native.read_bytes()
        adjudication = self.adjudication('redraw')

        async def inspect(*args, **kwargs):
            self.change_body()
            return adjudication

        with patch.object(REBUILD, 'ask', side_effect=inspect), \
                patch.object(REBUILD, 'generate_image_requests', new_callable=AsyncMock) as provider:
            asyncio.run(REBUILD.repair_images([self.page], 1))
        provider.assert_not_called()
        self.assertEqual(native.read_bytes(), old_bytes)
        attempt = next((self.folder / 'image-repairs/cover').glob('*/attempt.json'))
        self.assertEqual(REBUILD.read_json(attempt)['status'], 'stale')

    def test_repair_discards_adjudication_when_prompt_changes_during_await(self):
        native = self.add_image(False)
        old_bytes = native.read_bytes()
        adjudication = self.adjudication('redraw')

        async def inspect(*args, **kwargs):
            article = REBUILD.read_json(self.folder / 'reviewed.json')
            article['images'][0]['prompt'] += '\n独立审稿更新的配图说明。'
            REBUILD.write_json(self.folder / 'reviewed.json', article)
            return adjudication

        with patch.object(REBUILD, 'ask', side_effect=inspect), \
                patch.object(REBUILD, 'generate_image_requests', new_callable=AsyncMock) as provider:
            asyncio.run(REBUILD.repair_images([self.page], 1))
        provider.assert_not_called()
        self.assertEqual(native.read_bytes(), old_bytes)
        self.assertIn('独立审稿更新', REBUILD.read_json(self.folder / 'reviewed.json')['images'][0]['prompt'])

    def test_repair_discards_generated_image_when_current_asset_changes_during_await(self):
        native = self.add_image(False)
        adjudication = self.adjudication('redraw')
        generated = self.folder / 'candidate.png'
        generated.write_bytes(b'stale-repair-candidate')

        async def generate(requests, concurrency):
            self.save_image(b'newer-independently-generated-image')
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}]}

        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication), \
                patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            asyncio.run(REBUILD.repair_images([self.page], 1))
        self.assertEqual(native.read_bytes(), b'newer-independently-generated-image')
        self.assertEqual(REBUILD.read_json(self.folder / 'reviewed.json')['images'][0]['prompt'],
                         self.article['images'][0]['prompt'])
        attempt = next((self.folder / 'image-repairs/cover').glob('*/attempt.json'))
        self.assertEqual(REBUILD.read_json(attempt)['status'], 'stale')

    def test_repair_discards_generated_image_when_article_changes_during_await(self):
        native = self.add_image(False)
        old_bytes = native.read_bytes()
        adjudication = self.adjudication('redraw')
        generated = self.folder / 'candidate.png'
        generated.write_bytes(b'stale-article-repair')

        async def generate(requests, concurrency):
            self.change_body()
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}]}

        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication), \
                patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            asyncio.run(REBUILD.repair_images([self.page], 1))
        self.assertEqual(native.read_bytes(), old_bytes)
        self.assertIn('最终修订的事实', REBUILD.read_json(self.folder / 'reviewed.json')['body_markdown'])
        self.assertIsNone(REBUILD.current_visual_review(self.page))

    def test_unresolved_native_review_has_a_persistent_retry_cap(self):
        self.add_image(False)
        adjudication = self.adjudication('unresolved')
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=adjudication) as ask:
            asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=1))
            remaining = asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=1))
        ask.assert_awaited_once()
        self.assertTrue(remaining[0]['exhausted'])
        self.assertFalse(REBUILD.media_passed(self.page))

    def test_redraw_budget_survives_new_prompts_and_new_native_hashes(self):
        self.add_image(False)
        generation_count = 0

        async def inspect(*args, **kwargs):
            value = self.adjudication('redraw')
            value['corrected_prompt'] += str(generation_count)
            return value

        async def generate(requests, concurrency):
            nonlocal generation_count
            generation_count += 1
            generated = self.folder / f'repair-provider-{generation_count}.png'
            generated.write_bytes(f'redraw-{generation_count}'.encode())
            return {'results': [{'id': requests[0].id, 'status': 'succeeded', 'path': str(generated)}]}

        with patch.object(REBUILD, 'ask', side_effect=inspect), \
                patch.object(REBUILD, 'make_image_request', FakeImageRequest), \
                patch.object(REBUILD, 'generate_image_requests', side_effect=generate):
            for _ in range(3):
                asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=3))
                self.add_review(False)
            remaining = asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=3))
        self.assertEqual(generation_count, 3)
        self.assertTrue(remaining[0]['exhausted'])
        self.assertFalse(REBUILD.media_passed(self.page))

    def test_interrupted_native_inspection_resumes_its_saved_response(self):
        self.add_image(False)
        adjudication = self.adjudication()

        async def interrupted(prompt, target, **kwargs):
            REBUILD.write_json(target, adjudication)
            raise asyncio.CancelledError()

        with patch.object(REBUILD, 'ask', side_effect=interrupted):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=1))
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock) as ask:
            asyncio.run(REBUILD.repair_images([self.page], 1, max_attempts=1))
        ask.assert_not_called()
        self.assertTrue(REBUILD.media_passed(self.page))
        self.assertEqual(len(list((self.folder / 'image-repairs/cover').glob('*/attempt.json'))), 1)

    def test_concurrent_repair_commands_share_no_active_adjudication(self):
        self.add_image(False)
        adjudication = self.adjudication()
        inspections = 0

        async def run():
            inspecting, release = asyncio.Event(), asyncio.Event()

            async def inspect(*args, **kwargs):
                nonlocal inspections
                inspections += 1
                inspecting.set()
                await release.wait()
                return adjudication

            with patch.object(REBUILD, 'ask', side_effect=inspect):
                first = asyncio.create_task(REBUILD.repair_images([self.page], 1))
                await inspecting.wait()
                await REBUILD.repair_images([self.page], 1)
                release.set()
                await first

        asyncio.run(run())
        self.assertEqual(inspections, 1)
        self.assertTrue(REBUILD.media_passed(self.page))
        attempt = next((self.folder / 'image-repairs/cover').glob('*/attempt.json'))
        self.assertEqual(REBUILD.read_json(attempt)['status'], 'false_positive')

    def test_standalone_repair_reports_missing_current_inputs(self):
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock) as ask:
            with self.assertRaisesRegex(RuntimeError, 'Current polished articles'):
                asyncio.run(REBUILD.repair_failed_images([self.page], 1))
        ask.assert_not_called()
        report = REBUILD.read_json(self.out / 'image-repair-failures.json')
        self.assertEqual(report[0]['path'], self.page['path'])

    def test_standalone_repair_completes_native_resolution(self):
        self.add_image(False)
        with patch.object(REBUILD, 'ask', new_callable=AsyncMock, return_value=self.adjudication()) as ask:
            asyncio.run(REBUILD.repair_failed_images([self.page], 1))
        ask.assert_awaited_once()
        self.assertTrue(REBUILD.media_passed(self.page))
        self.assertEqual(REBUILD.read_json(self.out / 'image-repair-failures.json'), [])

    def test_polish_does_not_overwrite_concurrent_article_edit(self):
        revised = {key: self.article[key] for key in ['title', 'description', 'body_markdown']}
        revised['body_markdown'] += '\n主编旧版本的改写。'

        async def inspect(*args, **kwargs):
            self.change_body()
            return revised

        with patch.object(REBUILD, 'ask', side_effect=inspect), \
                patch.object(REBUILD, 'validate_article', return_value=[]):
            asyncio.run(REBUILD.polish_pages([self.page], 1))
        current = REBUILD.read_json(self.folder / 'reviewed.json')
        self.assertIn('最终修订的事实', current['body_markdown'])
        self.assertNotIn('主编旧版本的改写', current['body_markdown'])

    def test_watch_media_waits_for_pass_and_requeues_existing_stale_png(self):
        self.add_image(False)
        calls = {'repair': 0, 'generation': 0}
        real_sleep = asyncio.sleep

        async def sleep(_):
            await real_sleep(0)

        async def repair(pages, concurrency, **kwargs):
            calls['repair'] += 1
            if calls['repair'] == 1:
                article = REBUILD.read_json(self.folder / 'reviewed.json')
                article['images'][0]['prompt'] += '\n修复后的图解任务'
                REBUILD.write_json(self.folder / 'reviewed.json', article)
            return REBUILD.remaining_media(pages)

        async def generate(pages, concurrency, **kwargs):
            calls['generation'] += 1
            self.save_image(b'fresh-regenerated-image')

        async def review(pages, concurrency, **kwargs):
            if calls['generation'] and REBUILD.media_snapshot(self.page) is not None:
                self.add_review(True)

        async def run():
            await asyncio.wait_for(REBUILD.watch_media([self.page], 1), timeout=2)

        with patch.object(REBUILD, 'repair_images', side_effect=repair), \
                patch.object(REBUILD, 'images', side_effect=generate), \
                patch.object(REBUILD, 'image_qa', side_effect=review), \
                patch.object(REBUILD.asyncio, 'sleep', side_effect=sleep):
            asyncio.run(run())
        self.assertEqual(calls['generation'], 1)
        self.assertTrue(REBUILD.media_passed(self.page))

    def test_watch_media_reports_exhausted_failures_and_stops_both_loops(self):
        self.add_image(False)
        with patch.object(REBUILD, 'image_qa', new_callable=AsyncMock), \
                patch.object(REBUILD, 'repair_images', new_callable=AsyncMock,
                             return_value=[{'path': self.page['path'], 'exhausted': True}]):
            with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                asyncio.run(REBUILD.watch_media([self.page], 1))

    def test_status_distinguishes_saved_and_current_reviews(self):
        self.add_image()
        self.change_body()
        with patch('builtins.print'):
            counts = REBUILD.status([self.page])
        self.assertEqual(counts['polished_current'], 1)
        self.assertEqual(counts['visual'], 1)
        self.assertEqual(counts['visual_current'], 0)
        self.assertEqual(counts['visual_stale'], 1)
        self.assertEqual(counts['media_pending'], 1)


if __name__ == '__main__':
    unittest.main()
