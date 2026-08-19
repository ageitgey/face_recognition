from face2ai_app.adapters.null_expression import NullExpressionEngine
from face2ai_app.domain.models import FaceBox


def test_null_engine_is_unavailable_and_returns_none_per_box():
    engine = NullExpressionEngine("not installed")
    assert engine.available is False and engine.availability_reason == "not installed"
    assert engine.analyze(b"jpeg", [FaceBox(top=0, right=1, bottom=1, left=0)]) == [None]
