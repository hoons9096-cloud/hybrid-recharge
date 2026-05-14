import unittest


class CoreImportTests(unittest.TestCase):
    def test_import_core_module(self):
        import core_sim_v27  # noqa: F401


if __name__ == "__main__":
    unittest.main()
