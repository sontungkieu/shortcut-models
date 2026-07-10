import unittest

from fid_repeat_utils import parse_eval_fid_seeds, summarize_fid_repeat_records
from scripts.analyze_gmm_tide_fid_repeats import analyze_rows


class FidRepeatUtilsTest(unittest.TestCase):
    def test_parse_eval_fid_seeds_deduplicates_in_order(self):
        self.assertEqual(parse_eval_fid_seeds("101, 202,101,303"), [101, 202, 303])

    def test_parse_eval_fid_seeds_rejects_empty_or_negative(self):
        with self.assertRaises(ValueError):
            parse_eval_fid_seeds("")
        with self.assertRaises(ValueError):
            parse_eval_fid_seeds("1,-2")

    def test_summarize_fid_repeat_records_uses_sample_std(self):
        summary = summarize_fid_repeat_records(
            [
                {"metric_name": "fid/timesteps/128", "eval_seed": 1, "value": 7.0},
                {"metric_name": "fid/timesteps/128", "eval_seed": 2, "value": 7.2},
            ]
        )["metrics"]["fid/timesteps/128"]
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 7.1)
        self.assertAlmostEqual(summary["sample_std"], 2 ** 0.5 / 10)


class FidRepeatAnalysisTest(unittest.TestCase):
    @staticmethod
    def _jobs():
        jobs = {}
        for family in ("C0", "C4"):
            for training_seed in (0, 1):
                run_name = f"{family.lower()}-s{training_seed}"
                jobs[run_name] = {
                    "run_name": run_name,
                    "candidate_family": family,
                    "training_seed": training_seed,
                    "resume_kernel_ref": f"owner/{run_name}",
                    "eval_fid_seeds": "1,2,3",
                    "eval_fid_generations": 50048,
                }
        return jobs

    @staticmethod
    def _rows(delta):
        rows = []
        for family in ("C0", "C4"):
            for training_seed in (0, 1):
                run_name = f"{family.lower()}-s{training_seed}"
                for eval_seed, offset in zip((1, 2, 3), (-0.01, 0.0, 0.01)):
                    value = 7.2 + 0.05 * training_seed + offset
                    if family == "C4":
                        value += delta
                    rows.append(
                        {
                            "run_name": run_name,
                            "eval_seed": eval_seed,
                            "value": value,
                            "step": 400000,
                            "eval_fid_generations": 50048,
                        }
                    )
        return rows

    def test_measurement_gate_passes_consistent_practical_gain(self):
        analysis = analyze_rows(self._jobs(), self._rows(delta=-0.2))
        self.assertTrue(analysis["all_training_seeds_favor_c4"])
        self.assertTrue(analysis["measurement_gate_passed"])
        self.assertAlmostEqual(analysis["paired_eval_delta_c4_minus_c0"]["mean"], -0.2)

    def test_measurement_gate_rejects_subthreshold_gain(self):
        analysis = analyze_rows(self._jobs(), self._rows(delta=-0.05))
        self.assertTrue(analysis["all_training_seeds_favor_c4"])
        self.assertFalse(analysis["measurement_gate_passed"])


if __name__ == "__main__":
    unittest.main()
