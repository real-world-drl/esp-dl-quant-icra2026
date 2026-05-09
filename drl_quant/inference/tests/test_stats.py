"""Sanity tests for InferenceStats."""

import math

import pytest

from drl_quant.inference.stats import EpisodeStats, InferenceStats, _percentile


def test_percentile_handles_empty_and_single():
    assert math.isnan(_percentile([], 50))
    assert _percentile([42], 50) == 42
    assert _percentile([42], 99) == 42


def test_percentile_interpolates():
    xs = list(range(1, 11))  # 1..10
    # 50th percentile of 1..10 -> halfway between 5 and 6 -> 5.5
    assert _percentile(xs, 50) == pytest.approx(5.5)
    assert _percentile(xs, 0) == 1
    assert _percentile(xs, 100) == 10


def test_summary_aggregates_across_episodes():
    stats = InferenceStats()
    stats.record_episode(EpisodeStats(0, reward=10.0, steps=100, wall_seconds=5.0,
                                      inference_times_us=[10, 20, 30]))
    stats.record_episode(EpisodeStats(1, reward=12.0, steps=100, wall_seconds=4.0,
                                      inference_times_us=[40, 50, 60]))
    s = stats.summary()
    assert s['episodes'] == 2
    assert s['mean_reward'] == pytest.approx(11.0)
    assert s['mean_inference_us'] == pytest.approx(35.0)
    assert s['min_inference_us'] == 10
    assert s['max_inference_us'] == 60


def test_episode_fps_and_stdev():
    ep = EpisodeStats(0, reward=1.0, steps=200, wall_seconds=10.0,
                      inference_times_us=[100, 100, 200, 200])
    assert ep.fps == pytest.approx(20.0)
    # stdev of [100,100,200,200] = sqrt(100^2 * 4/3) ≈ 57.74
    assert ep.stdev_inference_us == pytest.approx(57.735, rel=1e-3)


def test_sqlite_log_round_trip(tmp_path):
    stats = InferenceStats(output_dir=tmp_path)
    stats.record_episode(EpisodeStats(0, reward=1.5, steps=3, wall_seconds=2.0,
                                      inference_times_us=[100, 200, 300]))
    stats.close()

    import sqlite3
    db = sqlite3.connect(tmp_path / 'inference_times.db')
    cur = db.cursor()
    rows = cur.execute(
        'SELECT episode_no, step, inference_time_us FROM inference_times ORDER BY step'
    ).fetchall()
    assert rows == [(0, 0, 100), (0, 1, 200), (0, 2, 300)]

    eps = cur.execute('SELECT episode_no, reward, steps FROM episodes').fetchall()
    assert eps == [(0, 1.5, 3)]
    db.close()
