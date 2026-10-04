#!/usr/bin/env python3
"""Standard-library tests for the project storage path guard."""

from __future__ import annotations

import unittest

from check_storage_path import is_under_ssd8


class StoragePathPolicyTests(unittest.TestCase):
    def test_rejects_ssd8_root(self) -> None:
        self.assertTrue(is_under_ssd8("/ssd8"))

    def test_rejects_ssd8_descendant(self) -> None:
        self.assertTrue(is_under_ssd8("/ssd8/cactus8603/new-result"))

    def test_rejects_parent_traversal_into_ssd8(self) -> None:
        self.assertTrue(is_under_ssd8("/ssd7/../ssd8/cactus8603"))

    def test_allows_project_repository(self) -> None:
        self.assertFalse(is_under_ssd8("/ssd1/cactus8603/roco_main/experiments"))

    def test_allows_large_data_destination(self) -> None:
        self.assertFalse(is_under_ssd8("/ssd6/cactus8603/ro_co/track2/assets"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
