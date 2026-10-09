import unittest
from solomon_dynamic.baseline_planner import compatible


class BaselineTests(unittest.TestCase):
    def test_capability_subset(self):
        uav = {"capabilities": ["payload", "camera"]}
        self.assertTrue(compatible(uav, {"required_capabilities": ["camera"]}))
        self.assertFalse(compatible(uav, {"required_capabilities": ["emergency"]}))


if __name__ == "__main__":
    unittest.main()
