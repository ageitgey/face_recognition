"""Static shell guards.

These tests protect the dependency-free browser shell described in ADR-001 and
docs/UI_DIRECTION.md: every asset referenced by index.html is served, the ES
module graph resolves, nothing is loaded from the network, and the enrollment
flow keeps its explicit-consent step and native <dialog> surfaces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "face2ai_app" / "static"
INDEX = STATIC / "index.html"
JS_DIR = STATIC / "js"

ASSET_REF = re.compile(r"""(?:src|href)=["'](/assets/[^"']+)["']""")
IMPORT_REF = re.compile(r"""from\s+["'](\.\/[^"']+)["']""")
EXTERNAL_REF = re.compile(r"""(?:https?:)?//(?!127\.0\.0\.1|localhost|www\.w3\.org/)[a-z0-9.-]+\.[a-z]{2,}""", re.I)


def _asset_refs() -> list[str]:
    return ASSET_REF.findall(INDEX.read_text(encoding="utf-8"))


def _js_files() -> list[Path]:
    return sorted(JS_DIR.glob("*.js"))


def _js_function_body(source: str, signature: str) -> str:
    """Text of a top-level function in one of the shell's modules.

    A source assertion, not a behavioural test: app.js is DOM-bound (it reads `document` at import
    time), so the rules below cannot be exercised with `node --test` without extracting the whole
    orchestrator. What *is* extractable already lives in model.js and is tested there; what remains
    here is where an assignment sits relative to an `await` — a question about the source itself.
    Top-level functions in these modules close with a `}` in the first column.
    """
    assert signature in source, f"{signature} not found"
    start = source.index(signature)
    end = source.index("\n}\n", start)
    return source[start:end]


def test_index_is_served_and_declares_module_entry(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'type="module"' in response.text
    assert "/assets/js/app.js" in response.text


def test_every_asset_referenced_by_index_is_served(client):
    refs = _asset_refs()
    assert refs, "index.html references no /assets/ files"
    for ref in refs:
        assert client.get(ref).status_code == 200, ref


def test_es_module_import_graph_resolves():
    files = _js_files()
    assert files, "no ES modules found"
    for file in files:
        for target in IMPORT_REF.findall(file.read_text(encoding="utf-8")):
            assert (JS_DIR / target).is_file(), f"{file.name} imports missing {target}"


@pytest.mark.parametrize("path", [INDEX, STATIC / "css" / "app.css", *_js_files()])
def test_static_shell_loads_nothing_from_the_network(path: Path):
    text = path.read_text(encoding="utf-8")
    hits = [m.group(0) for m in EXTERNAL_REF.finditer(text)]
    assert not hits, f"{path.name} references external hosts: {hits}"


def test_enrollment_uses_native_dialog_with_required_consent():
    html = INDEX.read_text(encoding="utf-8")
    assert re.search(r"<dialog[^>]*id=\"enrollDialog\"", html)
    consent = re.search(r"<input[^>]*id=\"consent\"[^>]*>", html)
    assert consent, "enrollment dialog has no consent control"
    assert 'type="checkbox"' in consent.group(0)
    assert "required" in consent.group(0)


def test_destructive_actions_do_not_use_blocking_browser_prompts():
    for file in _js_files():
        text = file.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:window\.)?(?:confirm|alert|prompt)\(", text), file.name
    html = INDEX.read_text(encoding="utf-8")
    assert re.search(r"<dialog[^>]*id=\"eraseDialog\"", html)


def test_shell_never_presents_distance_as_confidence():
    for file in [INDEX, *_js_files()]:
        text = file.read_text(encoding="utf-8").lower()
        assert "confidence" not in text, file.name


def test_consent_is_enforced_by_client_and_sent_to_the_api():
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    api_js = (JS_DIR / "api.js").read_text(encoding="utf-8")
    assert "if (!consent)" in app_js, "client-side consent guard missing"
    assert re.search(r"consent:\s*consent \? 'true' : 'false'", api_js), "consent flag not sent as query param"


def test_enrollment_dialog_is_reset_on_every_close_not_only_on_success():
    """A dialog closed with Escape after a failed enrollment must leave nothing behind.

    Measured before the fix: name and consent were cleared only on the success path, so the next
    LEARN click opened the dialog with the previous person's name and a pre-ticked biometric consent
    box — one click enrolled a *different* person as `display_name=Alice&consent=true`.
    """
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    closed = _js_function_body(app_js, "function onEnrollDialogClosed()")
    assert "resetEnrollForm()" in closed, "every close path (submit, cancel, Escape) must reset the form"
    reset = _js_function_body(app_js, "function resetEnrollForm()")
    assert "els.displayName.value = ''" in reset, "the display name survives the close"
    assert "els.consent.checked = false" in reset, "consent is biometric consent: it must never be pre-ticked"
    assert "state.enrollFrozen = null" in reset, "the frozen frame must not outlive the dialog"
    # One owner: the success path must not keep a second copy of the reset that can drift from this one.
    assert app_js.count("els.consent.checked = false") == 1
    assert app_js.count("els.displayName.value = ''") == 1


def test_the_frame_and_the_event_it_produced_are_published_as_one_pair():
    """A LEARN click during an in-flight request must not freeze frame N+1 with frame N's event.

    The loop runs every 450 ms and a recognize call takes ~160 ms, so a click landing inside that
    window used to freeze the newest frame together with the previous frame's event — the dialog then
    decided "one unknown face" from one frame and enrolled a different one.
    """
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    tick = _js_function_body(app_js, "async function tick()")
    assert "state.latest" not in tick, "tick() must publish nothing while its own request is in flight"
    handled = _js_function_body(app_js, "function handleRecognition(blob, event)")
    assert re.search(r"state\.latest = \{ blob, event \}", handled), "frame and event are one assignment"
    opened = _js_function_body(app_js, "function openEnrollDialog()")
    assert "state.enrollFrozen = state.latest" in opened, "the dialog freezes the pair, not two fields"
    for split in ("state.lastBlob", "state.lastEvent", "state.enrollBlob", "state.enrollEvent"):
        assert split not in app_js, f"{split}: a frame and its event must not live in separate fields"


def test_the_live_event_stream_recovers_from_a_connection_left_closed():
    """Per the EventSource spec a non-2xx status or a wrong content-type *fails* the connection and
    leaves `readyState` CLOSED; only a dropped connection reconnects on its own. Reasoned from the
    spec, not from a browser session — the client's own behaviour under both failures is pinned
    against a fake EventSource in tests/js/events.test.mjs.
    """
    events_js = (JS_DIR / "events.js").read_text(encoding="utf-8")
    assert "readyState" in events_js, "the two failure kinds are only distinguishable by readyState"
    assert "EventSource reconnects by itself" not in events_js, "that comment is true only for a dropped connection"
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    assert "onOpen:" in app_js, "the shell must notice when the stream comes back"
    assert "describeEventsStatus(" in app_js, "the stream state belongs in the Context card, not only in the log"
    html = INDEX.read_text(encoding="utf-8")
    row = re.search(r"<div class=\"context-row\"><span>Live events</span><span id=\"eventsContext\">([^<]*)</span></div>", html)
    assert row, "the Context card has no row for the live event stream"


def test_dialog_default_submit_buttons_are_the_primary_actions():
    """Implicit submission (Enter in a text field) must never trigger Cancel."""
    html = INDEX.read_text(encoding="utf-8")
    for form_id, expected in (("enrollForm", "learn"), ("eraseForm", "erase")):
        form = re.search(rf"<form[^>]*id=\"{form_id}\"[^>]*>(.*?)</form>", html, re.S)
        assert form, form_id
        submits = re.findall(r"<button[^>]*type=\"submit\"[^>]*>", form.group(1))
        assert submits, form_id
        assert f'value="{expected}"' in submits[0], f"{form_id}: first submit button is {submits[0]}"
        for cancel in re.findall(r"<button[^>]*>Cancel</button>", form.group(1)):
            assert 'type="button"' in cancel, cancel


def test_closed_dialogs_are_not_laid_out():
    """No author rule may set `display` on a dialog outside an [open] selector (UA hides closed dialogs)."""
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = match.group(1).strip(), match.group(2)
        if not re.search(r"(^|[\s,])(dialog|\.drawer|\.modal-card)(?![\w-])", selector):
            continue
        if "display" in body:
            assert "[open]" in selector, f"{selector!r} sets display without [open]"


# ---------- expression (Stage 1): opt-in, hedged, never a certainty ----------


def test_expression_toggle_and_tile_are_present_and_off_by_default():
    html = INDEX.read_text(encoding="utf-8")
    button = re.search(r"<button[^>]*id=\"expressionButton\"[^>]*>", html)
    assert button, "expression toggle button missing"
    assert "disabled" in button.group(0), "toggle must start disabled until /api/status says the engine is available"
    assert re.search(r"id=\"expressionTile\"", html), "expression metric tile missing"
    assert re.search(r"id=\"expressionValue\"[^>]*>off<", html), "tile must start as 'off' (opt-in default off)"
    for bar in ("valenceFill", "arousalFill"):
        assert re.search(rf"class=\"mood-bar-fill\" id=\"{bar}\"", html), bar
    assert "Expression: off" in html
    assert "a hint, not a fact" in html, "the tile must say what it is: a hint, never a fact"


def test_expression_wording_is_hedged_and_never_a_certainty(client):
    """The served bundle speaks of how a face *appears* ("looks …" — the shell is English); nothing reads as a finding.

    model.js keeps both wording tables (de "wirkt …" / en "looks …") so the view-model stays bilingual;
    app.js selects English. Forbidden anywhere in the shell: "is happy", "ist fröhlich", "erkannt", and
    "detected" applied to a mood/expression (faces may be detected; moods only ever *look* like something).
    """
    model_js = client.get("/assets/js/model.js").text
    assert "wirkt " in model_js
    assert "looks " in model_js
    assert "ist fröhlich" not in model_js
    app_js = client.get("/assets/js/app.js").text
    assert "EXPRESSION_LANG = 'en'" in app_js, "the browser shell is English; the tile must say 'looks …'"
    assert "looks " in app_js
    assert "expressionButton" in app_js
    assert "Expression: off" in app_js and "Expression: on" in app_js
    assert "Not available · " in app_js
    for path in [INDEX, STATIC / "css" / "app.css", *_js_files()]:
        text = path.read_text(encoding="utf-8").lower()
        assert "erkannt" not in text, f"{path.name}: expression must never be presented as 'erkannt'"
        assert not re.search(r"\bist (fröhlich|traurig|verärgert|ängstlich|überrascht|angewidert|abschätzig|neutral)\b", text), path.name
        assert not re.search(r"\b(is|are|was|were) (happy|sad|angry|fearful|surprised|disgusted|contemptuous)\b", text), (
            f"{path.name}: expression must be hedged ('looks happy'), never asserted ('is happy')"
        )
        assert not re.search(r"\b(mood|expression|emotion)s? (is |was |)detected\b|\bdetected (mood|expression|emotion)", text), (
            f"{path.name}: a mood is never 'detected' — it only ever 'looks' like something"
        )


def test_expression_toggle_is_the_only_thing_the_browser_sends_for_the_feature():
    api_js = (JS_DIR / "api.js").read_text(encoding="utf-8")
    assert re.search(r"'/api/expression'", api_js)
    assert re.search(r"JSON\.stringify\(\{ enabled: enabled === true \}\)", api_js), "toggle body must be exactly {enabled: bool}"
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    assert "api.setExpression(" in app_js
    assert "expression_available" in app_js and "expression_reason" in app_js and "expression_enabled" in app_js


def test_shell_and_assets_must_be_revalidated_by_the_browser(client):
    """A redeploy must never pair a fresh app.js with a heuristically cached model.js (real case, 2026-08-18)."""
    assert client.get("/").headers["cache-control"] == "no-cache"
    for path in ("/assets/js/app.js", "/assets/js/model.js", "/assets/css/app.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache", path
        assert "etag" in response.headers, path


# ---------- expression dynamics (Stage 2): server-fed mood/action entries, own-frame sparkline ----------


def test_valence_sparkline_is_inline_svg_hidden_until_there_is_data():
    html = INDEX.read_text(encoding="utf-8")
    svg = re.search(r"<svg[^>]*id=\"valenceSpark\"[^>]*>", html)
    assert svg, "expression tile has no valence sparkline"
    tag = svg.group(0)
    assert "hidden" in tag, "sparkline must start hidden (it needs at least two samples)"
    assert 'role="img"' in tag and "aria-label=" in tag, "sparkline needs an accessible name"
    assert "xmlns=" not in tag and "http" not in tag, "inline SVG must not reference any host"
    tile = re.search(r"<div[^>]*id=\"expressionTile\"[^>]*>(.*?)</section>", html, re.S)
    assert tile and 'id="valenceSpark"' in tile.group(1), "sparkline must live inside the expression tile"
    assert re.search(r"<polyline[^>]*id=\"valenceLine\"", html)


def test_browser_takes_mood_and_action_entries_from_the_server_stream():
    """The server's MoodTracker/ActionTracker are the single source of truth; the browser only renders them.

    The client-side `trackMood` debounce is gone: two consumers deriving two different moods from one
    stream would contradict each other. events.js is a same-origin EventSource on `?role=browser`
    (never `role=agent`, which would hand greetings to a voice agent).
    """
    events_js = JS_DIR / "events.js"
    assert events_js.is_file(), "static/js/events.js missing"
    events_text = events_js.read_text(encoding="utf-8")
    assert "new EventSource(" in events_text
    assert "/api/events?role=browser" in events_text
    assert "role=agent" not in events_text
    for kind in ("'mood'", "'action'"):
        assert f"addEventListener({kind}" in events_text, kind
    # The client's behaviour (subscribed URL, parsing, close(), the unsupported-browser branch) is pinned
    # against a fake EventSource in tests/js/events.test.mjs — not by a regex over its source here.
    app_js = (JS_DIR / "app.js").read_text(encoding="utf-8")
    assert re.search(r"import \{ subscribeEvents \} from '\./events\.js'", app_js)
    assert "subscribeEvents({" in app_js
    assert "onMood" in app_js and "onAction" in app_js and "onError" in app_js
    assert "describeAction(" in app_js and "pushSample(" in app_js and "sparklinePoints(" in app_js
    assert "Live events unavailable" in app_js
    # Replay discipline + log discipline (both pinned behaviourally in tests/js/model.test.mjs): a reconnect
    # replays the server buffer, and actions fire often enough to evict every other entry from the 8-slot log.
    assert "isFreshEntry(" in app_js, "replayed mood/action frames must not be logged as if they just happened"
    assert "allowActionEntry(" in app_js, "action entries need a display rate limit or they flush the event log"
    for path in _js_files():
        assert "trackMood" not in path.read_text(encoding="utf-8"), f"{path.name}: client-side mood debounce must be gone"


def test_action_wording_tables_exist_and_the_source_has_no_certainty_vocabulary(client):
    """A vocabulary tripwire over the shipped source text, not a proof of what the UI renders.

    It catches the phrasings we know read as findings ("is smiling", "smile detected", "micro-expression")
    anywhere in the static sources — comments included — and pins that both wording tables are there.
    It does not and cannot prove that every rendered label is hedged: that guard is behavioural and lives
    in tests/js/model.test.mjs ("describeAction never reads as a finding", 8 actions × 2 languages × 3 durations).
    """
    model_js = client.get("/assets/js/model.js").text
    for word in ("'brief '", "'held '", "'kurzes '", "'anhaltendes '", "brow_raise", "lip_press"):
        assert word in model_js, word
    assert "export function describeAction" in model_js
    assert "export function formatDuration" in model_js
    action_words = r"(smil|frown|brow|squint|eyes wide|nose wrinkl|lip press|lächel|stirnrunzel|brauen|nasenrümpf|lippen)"
    for path in [INDEX, STATIC / "css" / "app.css", *_js_files()]:
        text = path.read_text(encoding="utf-8").lower()
        assert not re.search(r"\b(is|are|was|were) (smiling|frowning|squinting|wrinkling|pressing)\b", text), (
            f"{path.name}: an action is a movement ('brief smile'), never a state ('is smiling')"
        )
        assert not re.search(rf"\bdetected\W+{action_words}|{action_words}\w*\W+(is |was |)detected\b", text), (
            f"{path.name}: an action is never 'detected'"
        )
        assert not re.search(r"\bmicro[- ]?expression", text), f"{path.name}: ~0.6 s resolution — never call these micro-expressions"
