"""A thin round's step (es2.Spec2.fixed_step, es2.step_scale): the step's scale over every member a round could have,
so a round where only some units land takes a step its share of the size instead of a full one out of fewer members.
Off, the field is left out of the spec and the scale is what it always was, to the bit."""
from __future__ import annotations

import unittest

from vibehash import es2

from tests.test_loop_stop import spec_json


class FixedStepTests(unittest.TestCase):
    def test_off_is_left_out_of_the_spec(self):
        s = es2.Spec2.from_json(spec_json())
        self.assertFalse(s.fixed_step)
        self.assertNotIn("fixed_step", s.to_json()["es"])  # a job's hash is what it was before the field
        self.assertEqual(s.to_json(), es2.Spec2.from_json(s.to_json()).to_json())

    def test_on_round_trips_and_is_checked(self):
        s = es2.Spec2.from_json(spec_json(fixed_step=True))
        self.assertTrue(s.fixed_step)
        self.assertIs(s.to_json()["es"]["fixed_step"], True)
        self.assertEqual(es2.Spec2.from_json(s.to_json()), s)
        self.assertFalse([p for p in s.problems_list() if "fixed_step" in p])
        for bad in (1, "yes", None):
            self.assertTrue([p for p in es2.Spec2.from_json(spec_json(fixed_step=bad)).problems_list()
                             if "fixed_step" in p], bad)

    def test_off_the_scale_is_the_old_one_to_the_bit(self):
        s = es2.Spec2.from_json(spec_json())
        for step in (0, 3, 11, 40):
            for members in (2, 6, 16):
                self.assertEqual(es2.step_scale(s, step, members), es2.lr_at(s, step) / members)

    def test_on_a_thin_round_takes_its_share_of_a_step(self):
        s = es2.Spec2.from_json(spec_json(fixed_step=True))  # 8 pairs: 16 members a full round
        full = es2.step_scale(s, 5, 16)
        self.assertEqual(full, es2.lr_at(s, 5) / 16)  # a full round's step is the same either way
        self.assertEqual(es2.step_scale(s, 5, 4), full)  # 2 of 8 pairs landed: the same scale, so a quarter the sum
        off = es2.Spec2.from_json(spec_json())
        self.assertEqual(es2.step_scale(off, 5, 4), 4 * full)  # (before: four times the scale on a quarter the pairs)


if __name__ == "__main__":
    unittest.main()
