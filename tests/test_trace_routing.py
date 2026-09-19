import numpy as np

from pomapf_env.trace_routing import bonus_rng, draw_tie_ranks


def test_rank_metadata_preserves_direct_two_draw_order_and_ties():
    rng = bonus_rng(42)
    reference = bonus_rng(42)
    ranks = draw_tie_ranks(rng, 17)
    first, second = reference.random((17, 5)), reference.random((17, 5))
    np.testing.assert_array_equal(ranks[:, 0].argmax(1), first.argmax(1))
    np.testing.assert_array_equal(ranks[:, 1].argmax(1), second.argmax(1))
    np.testing.assert_array_equal(rng.random(10), reference.random(10))
    assert ranks.dtype == np.float32
    np.testing.assert_array_equal(np.sort(ranks, axis=-1),
                                  np.broadcast_to(np.arange(5), (17, 2, 5)))


def test_restarting_tie_rng_reproduces_episode_not_previous_episode_state():
    expected = draw_tie_ranks(bonus_rng(123), 5)
    rng = bonus_rng(123)
    draw_tie_ranks(rng, 5)
    assert not np.array_equal(draw_tie_ranks(rng, 5), expected)
    np.testing.assert_array_equal(draw_tie_ranks(bonus_rng(123), 5), expected)
