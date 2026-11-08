"""Regression checks for applying a model's exact factual corrections safely."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('refine_content', Path(__file__).with_name('refine-content.py'))
REFINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFINE)


class FactualCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.article = {
            'body_markdown': '# 主题\n\n旧的具体说法。\n\n![解释关系](rebuild:cover)',
            'images': [{'id': 'cover', 'prompt': '原画面'}],
        }
        self.validation = patch.object(REFINE.REBUILD, 'validate_article', return_value=[])
        self.validation.start()
        self.addCleanup(self.validation.stop)

    def verdict(self, old='旧的具体说法。', new='加上条件后的准确说法。'):
        return {'passed': False, 'replacements': [{'old': old, 'new': new, 'reason': '原句缺少适用条件'}], 'images': []}

    def test_correction_preserves_source_and_image_anchors(self):
        result = REFINE.apply_replacements(self.article, self.verdict(), {})
        self.assertIn('加上条件后的准确说法。', result['body_markdown'])
        self.assertIn('旧的具体说法。', self.article['body_markdown'])
        self.assertIn('![解释关系](rebuild:cover)', result['body_markdown'])

    def test_ambiguous_replacement_is_rejected(self):
        self.article['body_markdown'] += '\n旧的具体说法。'
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            REFINE.apply_replacements(self.article, self.verdict(), {})

    def test_model_cannot_remove_or_relabel_an_image(self):
        with self.assertRaisesRegex(ValueError, 'anchors'):
            REFINE.apply_replacements(self.article, self.verdict(
                '![解释关系](rebuild:cover)', '![另一个关系](rebuild:cover)'), {})

    def test_auditor_cannot_fail_without_an_actionable_correction(self):
        with self.assertRaisesRegex(ValueError, 'concrete'):
            REFINE.apply_replacements(self.article, {'passed': False, 'replacements': [], 'images': []}, {})

    def test_illustration_only_factual_correction_is_actionable(self):
        prompt = '人物在白板前解释正文已经说明的计算条件与排序公式，图中的数值、单位和箭头与正文保持一致。' * 4
        result = REFINE.apply_replacements(self.article, {
            'passed': False, 'replacements': [], 'images': [{'id': 'cover', 'prompt': prompt}],
        }, {})
        self.assertEqual(result['body_markdown'], self.article['body_markdown'])
        self.assertEqual(result['images'][0]['prompt'], prompt)
        self.assertEqual(self.article['images'][0]['prompt'], '原画面')

    def test_illustration_only_correction_must_still_reference_an_existing_image(self):
        with self.assertRaisesRegex(ValueError, 'Invalid illustration correction'):
            REFINE.apply_replacements(self.article, {
                'passed': False, 'replacements': [], 'images': [{'id': 'missing', 'prompt': '准确画面' * 30}],
            }, {})


class FactualValidationTests(unittest.TestCase):
    def errors(self, statement):
        images = [{'id': key, 'alt': '解释关系', 'prompt': '具体画面和准确标签。' * 20}
                  for key in ['cover', 'fig-1', 'fig-2', 'fig-3', 'fig-4', 'fig-5', 'fig-6']]
        body = '# 题目\n\n' + '这段文字解释输入、处理条件和输出之间的关系。' * 80 + '\n' + statement + '\n'
        body += '\n'.join('![解释关系](rebuild:' + image['id'] + ')' for image in images)
        with patch.object(REFINE.REBUILD, 'read_json', return_value={'pages': []}):
            return REFINE.REBUILD.validate_article({'body_markdown': body, 'images': images},
                                                   {'kind': 'article', 'section': 'fundamentals', 'path': 'topic.md'})

    def test_legitimate_ratios_and_negative_guarantees_are_accepted(self):
        self.assertEqual(self.errors('在被选中的位置中，80% 替换为 MASK，10% 换随机词，10% 保留真实词。'), [])
        self.assertEqual(self.errors('按需加载不能保证100%可靠，也不意味着不会选错。'), [])

    def test_explicit_blanket_guarantee_is_rejected(self):
        self.assertTrue(any('无条件承诺' in error for error in self.errors('该流程保证100%可靠。')))


if __name__ == '__main__':
    unittest.main()
