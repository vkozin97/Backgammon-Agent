from pathlib import Path

from training.replay import ReplayBuffer
from training.replay_filter import filter_replay_matches


def test_filter_replay_matches_with_optional_filters(tmp_path: Path):
    replay = ReplayBuffer(storage_dir=str(tmp_path / "replay"))
    replay.add(
        state_vector=[0.0, 1.0],
        agent_id="trainable_0",
        opponent_id="trainable_1",
        game_id="m1",
        step_index=0,
        epoch=7,
        terminal_outcome=[1.0],
        match_length=42,
        match_agent_1_id="trainable_0",
        match_agent_2_id="trainable_1",
        match_number=0,
        game_number_in_match=2,
        final_dave_value=4,
        final_reward_value=2,
    )
    replay.add(
        state_vector=[0.0, 1.0],
        agent_id="trainable_2",
        opponent_id="trainable_3",
        game_id="m2",
        step_index=0,
        epoch=8,
        terminal_outcome=[1.0],
        match_length=12,
        match_agent_1_id="trainable_2",
        match_agent_2_id="trainable_3",
        match_number=1,
        game_number_in_match=1,
        final_dave_value=2,
        final_reward_value=1,
    )
    replay.close()

    base = str(tmp_path / "replay")
    assert filter_replay_matches(base) == [("m1", 2), ("m2", 1)]
    assert filter_replay_matches(base, epoch=7) == [("m1", 2)]
    assert filter_replay_matches(base, match_length=12, final_dave_value=2) == [("m2", 1)]
    assert filter_replay_matches(base, agent_1_id="trainable_0", agent_2_id="trainable_1", final_reward_value=2) == [("m1", 2)]
    assert filter_replay_matches(base, epoch=999) == []
