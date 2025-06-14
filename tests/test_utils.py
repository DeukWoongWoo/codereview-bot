import unittest
from utils import split_patch


class SplitPatchTests(unittest.TestCase):
    def test_split_patch_basic(self):
        patch = "a\n" * 10
        result = split_patch(patch, 5)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(len(chunk) <= 5 for chunk in result))

    def test_split_patch_zero(self):
        patch = "abc"
        self.assertEqual(split_patch(patch, 0), [patch])


if __name__ == "__main__":
    unittest.main()
