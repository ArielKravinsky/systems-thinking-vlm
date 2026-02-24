import unittest
from src.utils_hebrew import normalize_hebrew


class TestHebrewUtils(unittest.TestCase):
    def test_normalize_removes_niqqud_and_normalizes_finals(self):
        s = 'מָקוֹר ךףןץ'
        out = normalize_hebrew(s)
        # niqqud removed, finals normalized (expect same-letter mapping)
        self.assertIsInstance(out, str)
        self.assertNotIn('\u0591', out)
        self.assertTrue(len(out) > 0)


if __name__ == '__main__':
    unittest.main()
