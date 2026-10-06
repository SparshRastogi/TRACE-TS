import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from AtteFinalPipeline.attribution import combined, bottom10pct
from AtteFinalPipeline.attribution.engine import XAIEngine
from AtteFinalPipeline.attribution.progress import ProgressTracker
from AtteFinalPipeline.attribution.regions import (
    build_result_dict,
    extract_high_attribution_regions,
)
from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.data.loading import _get_dataset_config, load_data
from AtteFinalPipeline.data.sensors import get_null_class_indices, get_sensor_config
from AtteFinalPipeline.embeddings.encoders import compute_mantis_embeddings_batch
from AtteFinalPipeline.expert.dataset import SensorDataset
from AtteFinalPipeline.expert.model import create
from AtteFinalPipeline.expert.settings import get_args
from AtteFinalPipeline.tools.filter_subset import filter_subset
from trace.gt_gen.prompt import format_attention_data


class LinearClassifier(torch.nn.Module):
    def forward(self, x):
        feature = x.mean(dim=(1, 2))
        logits = torch.stack((-feature, feature), dim=1)
        return feature, logits, None


class ChannelEncoder:
    def transform(self, x):
        return np.mean(x, axis=-1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.names = ["Acc_X", "Acc_Y", "Acc_Z"]
        self.data = np.arange(60, dtype=np.float32).reshape(20, 3) / 60
        self.attribution = self.data.copy()
        self.result = build_result_dict(
            self.data,
            self.attribution,
            1,
            np.array([0.1, 0.9]),
            7,
            1,
            self.names,
            "train",
            ["still", "walking"],
        )

    def test_sensor_layouts_cover_every_channel(self):
        for dataset, channels in [
            ("ucihar", 9),
            ("uschad", 6),
            ("pamap2", 52),
            ("capture24", 3),
            ("mhealth", 23),
            ("shoaib", 45),
            ("opportunity", 79),
        ]:
            with self.subTest(dataset=dataset):
                names, groups = get_sensor_config(channels, dataset)
                self.assertEqual(len(names), channels)
                self.assertEqual(len(set(names)), channels)
                self.assertEqual(
                    sorted(index for _, indices, _ in groups for index in indices),
                    list(range(channels)),
                )

    def test_dataset_config_preserves_argv_and_corrected_labels(self):
        argv = sys.argv[:]
        config = _get_dataset_config("uschad")
        self.assertEqual(sys.argv, argv)
        self.assertEqual(config["class_map"], USCHAD_CLASS_MAP)
        self.assertEqual(config["class_map"][4], "Walking Downstairs")
        self.assertEqual(config["class_map"][10], "Elevator Up")
        self.assertEqual(get_null_class_indices(["Null", "walk"], "mhealth"), {0})
        self.assertEqual(get_null_class_indices(["walk"], "mhealth_nonull"), set())

    def test_processed_loader_preserves_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.stack([self.data, self.data + 2])
            np.savez(Path(directory) / "test.npz", data=values, target=[4, 9])
            loaded, labels = load_data("uschad", directory, "test")
            np.testing.assert_array_equal(loaded, values)
            np.testing.assert_array_equal(labels, [4, 9])

    def test_ucihar_uses_configured_path(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.stack([self.data + i for i in range(10)])
            np.save(Path(directory) / "X_train.npy", values)
            np.save(Path(directory) / "y_train.npy", [1, 2] * 5)
            _, config, _ = get_args(["--dataset", "ucihar", "--path_data", directory])
            with contextlib.redirect_stdout(io.StringIO()):
                train = SensorDataset(**config, prefix="train")
                val = SensorDataset(**config, prefix="val")
            np.testing.assert_array_equal(train.data, values[:8])
            np.testing.assert_array_equal(val.data, values[8:])
            self.assertEqual(train.target.tolist(), [0, 1] * 4)

    def test_regions_keep_boundary_runs_and_rank(self):
        values = np.zeros((20, 2), dtype=np.float32)
        values[:2, 0] = [9, 10]
        values[-2:, 1] = [8, 7]
        regions, threshold = extract_high_attribution_regions(values, ["x", "y"])
        self.assertGreater(threshold, 0)
        self.assertEqual(
            [(r["start_t"], r["end_t"]) for r in regions], [(0, 1), (18, 19)]
        )
        self.assertEqual([r["peak_timestep"] for r in regions], [1, 18])

    def test_bottom_regions_use_low_tail(self):
        values = np.arange(40, dtype=np.float32).reshape(20, 2)
        regions, threshold = bottom10pct.extract_low_attribution_regions(
            values, ["x", "y"]
        )
        self.assertEqual(sum(region["length"] for region in regions), 4)
        self.assertTrue(all(max(r["importance_values"]) <= threshold for r in regions))

    def test_attribution_matches_linear_classifier(self):
        batch = torch.tensor(np.stack([self.data + 1, self.data + 2]))
        baseline = torch.zeros((1, 20, 3))
        with contextlib.redirect_stdout(io.StringIO()):
            engine = XAIEngine(
                LinearClassifier(),
                baseline,
                baseline,
                torch.device("cpu"),
                use_compile=False,
            )
        targets = torch.ones(2, dtype=torch.long)
        expected = batch.numpy() / batch.numpy().max(axis=(1, 2), keepdims=True)
        ig = engine.compute_ig_batch(batch, targets, n_steps=5)
        shap = engine.compute_shap_batch(batch, targets)
        np.testing.assert_allclose(ig, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(shap, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            engine._combine_batch(ig, shap), expected, rtol=1e-5, atol=1e-6
        )

    def test_embedding_batches_preserve_channel_order(self):
        data = np.broadcast_to(np.array([1, 4, 9], dtype=np.float32), (5, 20, 3)).copy()
        embeddings = compute_mantis_embeddings_batch(
            ChannelEncoder(), data, torch.device("cpu"), batch_size=2
        )
        np.testing.assert_allclose(embeddings, [[1, 4, 9]] * 5)

    def test_serialized_evidence_is_accepted_by_teacher_prompt(self):
        entry = combined.build_sample_json(
            self.result,
            self.names,
            np.arange(6, dtype=np.float32),
            class_map=["still", "walking"],
        )
        prompt = format_attention_data(json.loads(json.dumps(entry)))
        self.assertIn("Predicted Activity: walking", prompt)
        self.assertIn("Signal t=", prompt)
        self.assertEqual(entry["mantis_embedding"]["embedding_dim"], 6)
        self.assertEqual(entry["sample_idx"], 7)

    def test_ablation_metadata_stays_distinct(self):
        methods = {}
        for name in ["combined", "ig_only", "shap_only", "random"]:
            module = importlib.import_module(f"AtteFinalPipeline.attribution.{name}")
            entry = module.build_sample_json(
                self.result, self.names, class_map=["still", "walking"]
            )
            methods[name] = entry["analysis_metadata"]["attribution_combination"][
                "method"
            ]
        self.assertEqual(len(set(methods.values())), 4)

    def test_deepconvlstm_import_does_not_change_combined_metadata(self):
        module = importlib.import_module("AtteFinalPipeline.attribution.deepconvlstm")
        entry = module.build_sample_json(
            self.result, self.names, class_map=["still", "walking"]
        )
        self.assertEqual(entry["analysis_metadata"]["model"], "DeepConvLSTM")
        entry = combined.build_sample_json(
            self.result, self.names, class_map=["still", "walking"]
        )
        self.assertEqual(entry["analysis_metadata"]["model"], "AttendDiscriminate")

    def test_expert_model_runs_on_cpu(self):
        _, _, config = get_args(["--dataset", "ucihar"])
        config.update(experiment="test_models", train_mode=False)
        with contextlib.redirect_stdout(io.StringIO()):
            model = create("AttendDiscriminate", config).eval()
        with torch.no_grad():
            _, logits, _ = model(torch.randn(2, 128, 9))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(model.centers.device.type, "cpu")

    def test_subset_filter_excludes_run_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["reference", "source"]:
                (root / name).mkdir()
                (root / name / "train_class1_walk_s0001.json").write_text("{}")
                (root / name / "progress.json").write_text("{}")
            (root / "reference" / "test_class1_walk_s0002.json").write_text("{}")
            copied, missing = filter_subset(
                root / "reference", root / "source", root / "out"
            )
            self.assertEqual(copied, 1)
            self.assertEqual(missing, ["test_class1_walk_s0002.json"])
            self.assertEqual(len(list((root / "out").iterdir())), 1)

    def test_split_saves_only_correct_nonnull_samples(self):
        values = np.stack(
            [
                np.ones((20, 3), dtype=np.float32),
                -np.ones((20, 3), dtype=np.float32),
                np.ones((20, 3), dtype=np.float32),
            ]
        )
        labels = np.array([1, 1, 0])
        baseline = torch.zeros((1, 20, 3))
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            model = LinearClassifier()
            engine = XAIEngine(
                model, baseline, baseline, torch.device("cpu"), use_compile=False
            )
            tracker = ProgressTracker(directory, {"train": 3})
            tracking = combined.process_split(
                "train",
                values,
                labels,
                model,
                None,
                engine,
                directory,
                self.names,
                [("acc", [0, 1, 2], self.names)],
                torch.device("cpu"),
                tracker,
                ["Null", "walking"],
                "ucihar",
                n_steps=5,
                plot_workers=1,
                plot_every_n=0,
                null_class_indices={0},
                xai_batch_size=2,
            )
            samples = list(Path(directory).glob("train_class*.json"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(json.loads(samples[0].read_text())["sample_idx"], 0)
            self.assertEqual(
                sum(
                    t["correct"] and not t.get("null_skipped", False) for t in tracking
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
