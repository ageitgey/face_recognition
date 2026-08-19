from __future__ import annotations

from datetime import datetime, timezone

import pytest

from face2ai_app.domain.models import Expression, MoodTransition
from face2ai_app.services.mood import MoodTracker

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def happy(v: float = 0.6) -> Expression:
    return Expression(dominant="Happiness", scores={"Happiness": 0.9, "Neutral": 0.1}, valence=v, arousal=0.1)


def sad() -> Expression:
    return Expression(dominant="Sadness", scores={"Sadness": 0.8, "Neutral": 0.2}, valence=-0.5, arousal=-0.2)


def test_mood_commits_after_stable_ticks_and_ignores_flicker():
    t = MoodTracker(stable_ticks=3, min_score=0.5)
    assert t.observe("KNOWN:a", happy(), T0) is None
    assert t.observe("KNOWN:a", sad(), T0) is None  # flicker resets the streak
    assert t.observe("KNOWN:a", happy(), T0) is None
    assert t.observe("KNOWN:a", happy(), T0) is None
    tr = t.observe("KNOWN:a", happy(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == (None, "Happiness") and tr.valence > 0
    assert tr.at == T0
    assert t.current() == ("Happiness", tr.valence, tr.arousal)


def test_mood_switch_and_reset_on_presence_change():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2):
        t.observe("KNOWN:a", happy(), T0)
    assert t.observe("KNOWN:a", sad(), T0) is None
    tr = t.observe("KNOWN:a", sad(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == ("Happiness", "Sadness")
    reset = t.observe("NO_FACE:", None, T0)
    assert reset is not None and (reset.from_mood, reset.to_mood) == ("Sadness", None)
    assert t.current() == (None, None, None)


def test_low_confidence_never_commits():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    weak = Expression(dominant="Neutral", scores={"Neutral": 0.4, "Happiness": 0.3, "Sadness": 0.3})
    assert all(t.observe("KNOWN:a", weak, T0) is None for _ in range(5))
    assert t.current() == (None, None, None)


def test_none_expression_for_stable_ticks_frames_resets_mood_exactly_once():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2):
        t.observe("KNOWN:a", happy(), T0)
    assert t.current()[0] == "Happiness"
    assert t.observe("KNOWN:a", None, T0) is None  # one missing frame is not enough
    reset = t.observe("KNOWN:a", None, T0)
    assert reset is not None and (reset.from_mood, reset.to_mood) == ("Happiness", None)
    assert t.current() == (None, None, None)
    assert t.observe("KNOWN:a", None, T0) is None  # already None: no second transition
    assert t.observe("KNOWN:a", None, T0) is None


def test_single_none_frame_keeps_mood():
    t = MoodTracker(stable_ticks=3, min_score=0.5)
    for _ in range(3):
        t.observe("KNOWN:a", happy(), T0)
    before = t.current()
    assert before[0] == "Happiness"
    assert t.observe("KNOWN:a", None, T0) is None
    assert t.current() == before
    assert t.observe("KNOWN:a", happy(), T0) is None  # same mood continues, no transition
    assert t.current() == before


def test_reset_returns_the_mood_end_and_clears_the_smoothing():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    assert t.reset() is None  # nothing set: nothing to announce
    for _ in range(2):
        t.observe("KNOWN:a", happy(), T0, identity_id="a", display_name="Ada")
    assert t.current()[0] == "Happiness"
    ended = t.reset(T0)
    assert ended is not None and (ended.from_mood, ended.to_mood) == ("Happiness", None)
    assert (ended.at, ended.identity_id, ended.display_name) == (T0, "a", "Ada")
    assert t.current() == (None, None, None)
    assert t.reset() is None  # already None: no second transition
    # EMA and streak are gone too: a fresh commit needs the full stable_ticks again.
    assert t.observe("KNOWN:a", happy(), T0) is None
    tr = t.observe("KNOWN:a", happy(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == (None, "Happiness")


def test_reset_without_mood_still_clears_the_streak():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    assert t.observe("KNOWN:a", happy(), T0) is None  # streak 1
    assert t.reset() is None
    assert t.observe("KNOWN:a", happy(), T0) is None  # streak restarted: not 2 yet
    tr = t.observe("KNOWN:a", happy(), T0)
    assert tr is not None and tr.to_mood == "Happiness"


def test_identity_propagates_onto_transitions():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    t.observe("KNOWN:a", happy(), T0, identity_id="a", display_name="Ada")
    tr = t.observe("KNOWN:a", happy(), T0, identity_id="a", display_name="Ada")
    assert tr is not None and (tr.identity_id, tr.display_name) == ("a", "Ada")
    # The reset caused by a presence change is about the person whose mood ends, not the newcomer.
    reset = t.observe("KNOWN:b", happy(), T0, identity_id="b", display_name="Bo")
    assert reset is not None and reset.to_mood is None
    assert (reset.identity_id, reset.display_name) == ("a", "Ada")


def test_missing_valence_and_arousal_stay_none():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    bare = Expression(dominant="Happiness", scores={"Happiness": 0.9})
    t.observe("KNOWN:a", bare, T0)
    tr = t.observe("KNOWN:a", bare, T0)
    assert tr is not None and tr.to_mood == "Happiness"
    assert (tr.valence, tr.arousal) == (None, None)
    assert t.current() == ("Happiness", None, None)


def test_key_change_and_return_needs_a_fresh_full_streak():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2):
        t.observe("KNOWN:a", happy(), T0)
    assert t.current()[0] == "Happiness"
    ended = t.observe("NO_FACE:", None, T0)  # key change ends the mood
    assert ended is not None and (ended.from_mood, ended.to_mood) == ("Happiness", None)
    # Same key again: EMA and streak restarted, so the first frame cannot commit ...
    assert t.observe("KNOWN:a", happy(), T0) is None
    assert t.current() == (None, None, None)
    # ... only the full streak does.
    tr = t.observe("KNOWN:a", happy(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == (None, "Happiness")


def test_candidate_flip_back_resets_the_streak():
    t = MoodTracker(stable_ticks=3, min_score=0.5)
    assert t.observe("KNOWN:a", happy(), T0) is None  # Happiness streak 1
    assert t.observe("KNOWN:a", happy(), T0) is None  # Happiness streak 2
    assert t.observe("KNOWN:a", sad(), T0) is None  # candidate flips to Sadness (streak 1)
    assert t.observe("KNOWN:a", happy(), T0) is None  # back to Happiness: streak 1 again, not 3
    assert t.observe("KNOWN:a", happy(), T0) is None  # streak 2
    assert t.current() == (None, None, None)
    tr = t.observe("KNOWN:a", happy(), T0)  # streak 3 -> commit
    assert tr is not None and tr.to_mood == "Happiness"


def test_expression_without_scores_counts_as_missing_frame():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2):
        t.observe("KNOWN:a", happy(), T0)
    assert t.current()[0] == "Happiness"
    blank = Expression(dominant="Anger", scores={})  # dominant is ignored; no scores = nothing to smooth
    assert t.observe("KNOWN:a", blank, T0) is None
    assert t.current()[0] == "Happiness"  # one missing frame keeps the mood ...
    ended = t.observe("KNOWN:a", blank, T0)  # ... stable_ticks missing frames end it
    assert ended is not None and (ended.from_mood, ended.to_mood) == ("Happiness", None)
    assert t.observe("KNOWN:a", blank, T0) is None  # and "Anger" never becomes a mood
    assert t.current() == (None, None, None)


def test_transition_is_a_wire_model_without_biometrics():
    assert set(MoodTransition.model_fields) == {
        "at", "identity_id", "display_name", "from_mood", "to_mood", "valence", "arousal",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stable_ticks": 0},
        {"min_score": 0.0},
        {"min_score": 1.5},
        {"alpha": 0.0},
        {"alpha": 1.5},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        MoodTracker(**kwargs)


def test_affect_is_live_while_current_is_frozen():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2):
        tr = t.observe("KNOWN:a", happy(0.6), T0)
    assert tr is not None and t.current()[1] == tr.valence
    t.observe("KNOWN:a", happy(-0.2), T0)  # same mood, valence swings
    assert t.current()[1] == tr.valence  # frozen at commit
    # live EMA (alpha 0.5, zero start) over valence 0.6, 0.6, -0.2 -> 0.3, 0.45, 0.125 and
    # arousal 0.1, 0.1, 0.1 -> 0.05, 0.075, 0.0875; affect() rounds to 3 decimals
    # (float accumulation gives 0.08750000000000001, so round-to-3 is 0.088)
    assert t.affect() == (pytest.approx(0.125), pytest.approx(0.088))
    assert t.affect() != (None, None)
    t.reset()
    assert t.affect() == (None, None)
