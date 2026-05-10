"""SqliteLogger tests — verify schema + buffered-flush semantics.

These don't drive an actual env; they exercise the logger directly so the
tests stay fast and deterministic.
"""

import sqlite3

import numpy as np
import pytest

from quaid_env.sqlite_logger import (
    SqliteLogger,
    _OBS_COLS, _ACTION_COLS, _REWARD_COLS, _THETA_COLS,
)


def _read_table(path, table):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(f'SELECT * FROM {table}').fetchall()
    finally:
        conn.close()


def _columns(path, table):
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(f'PRAGMA table_info({table})')
        return [row[1] for row in cur.fetchall()]
    finally:
        conn.close()


def test_creates_four_tables_with_expected_columns(tmp_path):
    """Schema must match ``QuaidLogging.cpp`` 1:1 — column names + order
    are what downstream analysis tooling expects."""
    log = SqliteLogger(tmp_path / 'run.sqlite')
    log.close()

    obs_cols = _columns(tmp_path / 'run.sqlite', 'observations')
    # PRAGMA includes the leading 'id' column.
    assert obs_cols[0] == 'id'
    assert tuple(obs_cols[1:]) == _OBS_COLS

    assert _columns(tmp_path / 'run.sqlite', 'actions')[1:] == list(_ACTION_COLS)
    assert _columns(tmp_path / 'run.sqlite', 'rewards')[1:] == list(_REWARD_COLS)
    assert _columns(tmp_path / 'run.sqlite', 'theta_updates')[1:] == list(_THETA_COLS)


def test_buffering_and_flush(tmp_path):
    """Rows aren't visible until flush_episode(); after flush they are.
    Mirrors the C++ pattern (buffer per episode, commit at done)."""
    path = tmp_path / 'run.sqlite'
    log = SqliteLogger(path)

    # Buffer one row of each kind.
    log.log_observation(
        episode_no=1, step=1, time=1000, time_delta=5,
        distance=0.6, voltage=12.0, current=2.0,
        yaw_delta=0.0, yaw_mean=0.05, yaw=0.05, pitch=0.0, roll=0.0,
        obs_age=0,
        servo0=100, servo1=110, servo2=120, servo3=130,
        servo4=140, servo5=150, servo6=160, servo7=170,
        current_front_left=0.5, current_front_right=0.4,
        current_back_left=0.3, current_back_right=0.2,
        acc_x=0.0, acc_y=0.0, acc_z=9.8, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        position_x=6.0, position_y=0.0, position_z=16.0, theta=0.0, done=0,
    )
    log.log_action(episode_no=1, step=1, time=1000, action=np.zeros(8))
    log.log_reward(
        episode_no=1, step=1, time=1000,
        reward=2.5, speed=2.0, distance=0.6,
        yaw_delta=0.0, yaw=0.5, pitch=0.0, roll=0.0, z_position=0.0,
        current_front_left=0.0, current_front_right=0.0,
        current_back_left=0.0, current_back_right=0.0,
        current=0.05, acc_z=-0.005, acc_x=0.0, acc_y=0.0,
        action_smoothness=-0.001,
    )

    # Before flush: no rows visible to a fresh reader.
    pre = sqlite3.connect(str(path))
    assert pre.execute('SELECT COUNT(*) FROM observations').fetchone() == (0,)
    pre.close()

    log.flush_episode()

    # After flush: each table has one row.
    assert len(_read_table(path, 'observations')) == 1
    assert len(_read_table(path, 'actions')) == 1
    assert len(_read_table(path, 'rewards')) == 1

    # Spot-check a few values made it through with the right types.
    obs_row = _read_table(path, 'observations')[0]
    # position_x is in column 32 (1-indexed after id=1, episode_no=2 ...);
    # easier: query by column name.
    conn = sqlite3.connect(str(path))
    try:
        px = conn.execute('SELECT position_x FROM observations').fetchone()[0]
        assert px == pytest.approx(6.0)
        servo3 = conn.execute('SELECT servo3 FROM observations').fetchone()[0]
        assert servo3 == 130
    finally:
        conn.close()
    log.close()


def test_action_dim_validation(tmp_path):
    """8-dim action is load-bearing — wrong dim must fail loudly rather
    than silently logging a partial row."""
    log = SqliteLogger(tmp_path / 'run.sqlite')
    with pytest.raises(ValueError, match='8-dim'):
        log.log_action(episode_no=1, step=1, time=0, action=[0.0] * 7)
    log.close()


def test_missing_observation_field_raises(tmp_path):
    """A typo in a kwarg must raise KeyError instead of silently inserting
    NULL — schema drift is hard to spot once the data is in."""
    log = SqliteLogger(tmp_path / 'run.sqlite')
    with pytest.raises(KeyError):
        log.log_observation(episode_no=1, step=1)  # missing all the rest
    log.close()


def test_close_flushes_pending(tmp_path):
    """Forgetting to flush_episode before close must not lose data."""
    path = tmp_path / 'run.sqlite'
    log = SqliteLogger(path)
    log.log_action(episode_no=1, step=1, time=0, action=np.zeros(8))
    log.close()  # no explicit flush_episode

    assert len(_read_table(path, 'actions')) == 1


def test_theta_update_round_trip(tmp_path):
    path = tmp_path / 'run.sqlite'
    log = SqliteLogger(path)
    log.log_theta_update(
        episode_no=1, step=10, time=12345,
        current_theta=0.1, yaw=0.05, yaw_mean=0.07,
        random=0.02, new_theta=0.12,
    )
    log.flush_episode()
    log.close()

    rows = _read_table(path, 'theta_updates')
    assert len(rows) == 1
    # row layout: (id, episode_no, step, time, current_theta, yaw, yaw_mean, random, new_theta)
    _, ep, step, t, cur, yaw, ym, rnd, new = rows[0]
    assert ep == 1 and step == 10 and t == 12345
    assert cur == pytest.approx(0.1)
    assert new == pytest.approx(0.12)
