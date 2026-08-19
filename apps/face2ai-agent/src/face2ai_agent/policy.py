"""Pure decision logic for the voice agent: what to say when presence changes, and how the
system prompt is built. No I/O here so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import AgentConfig
from .presence import PresenceMemory, StoreChange, Transition


@dataclass(frozen=True, slots=True)
class Prompt:
    """An instruction for the LLM (natural greeting) plus a fixed fallback sentence for TTS."""

    kind: str  # greet_known | greet_unknown | announce_multiple | welcome_enrolled | farewell
    instructions: str
    fallback_text: str
    display_name: str | None = None
    identity_id: str | None = None


MULTIPLE_ANNOUNCE_COOLDOWN_SECONDS = 60
MAX_TRANSITION_AGE_SECONDS = 20  # older transitions (replayed / delayed) are not spoken


@dataclass
class GreetingPolicy:
    """Decides which presence transitions deserve a spoken reaction.

    Transition-based: a known person is greeted when they arrive (a stable KNOWN transition), the
    same person again only after the server cooldown; unknown people and crowds are rate-limited.
    While an agent is subscribed the browser leaves spoken greetings to it and only logs the hand-off.
    """

    config: AgentConfig
    cooldown_seconds: int = 15
    last_known: dict[str, datetime] = field(default_factory=dict)  # last greeting per identity
    last_seen: dict[str, datetime] = field(default_factory=dict)  # last time a person was in front of the camera
    present_identity: str | None = None
    last_unknown_at: datetime | None = None
    last_multiple_at: datetime | None = None

    def _note_departure(self, now: datetime) -> None:
        if self.present_identity is not None:
            self.last_seen[self.present_identity] = now
        self.present_identity = None

    def on_transition(self, transition: Transition, now: datetime | None = None) -> Prompt | None:
        now = now or datetime.now(timezone.utc)
        lang = self.config.language
        if (now - transition.at).total_seconds() > MAX_TRANSITION_AGE_SECONDS:
            return None  # stale: the person may be long gone
        arriving = transition.identity_id if transition.to_state == "KNOWN" else None
        if arriving != self.present_identity:
            self._note_departure(now)
            self.present_identity = arriving
        if arriving:
            if not self.config.greet_known:
                return None
            last = self.last_known.get(arriving)
            if last is not None and (now - last).total_seconds() < self.cooldown_seconds:
                return None
            # A brief drop-out (recognition flicker, turned away, tab hidden for a moment) is not a
            # new arrival: greet again only after the person was really away for a while.
            seen = self.last_seen.get(arriving)
            if seen is not None and (now - seen).total_seconds() < self.config.regreet_after_seconds:
                self.last_seen[arriving] = now
                return None
            self.last_known[arriving] = now
            name = transition.display_name or "there"
            return Prompt(
                kind="greet_known",
                display_name=transition.display_name,
                identity_id=transition.identity_id,
                instructions=(
                    f"{name} just stepped in front of the camera and Face2AI recognized them. "
                    f"Greet {name} warmly by name in one short sentence in {lang}. "
                    "If you were mid-conversation with someone else, acknowledge the change briefly."
                ),
                fallback_text=_fallback_known(name, lang),
            )
        if transition.to_state == "UNKNOWN":
            if not self.config.greet_unknown:
                return None
            if (
                self.last_unknown_at is not None
                and (now - self.last_unknown_at).total_seconds() < self.config.unknown_greeting_cooldown_seconds
            ):
                return None
            self.last_unknown_at = now
            return Prompt(
                kind="greet_unknown",
                instructions=(
                    "Someone stepped in front of the camera whom Face2AI does not recognize. "
                    f"Say hello in one short sentence in {lang}, say you do not know them yet, and mention "
                    "that they can teach Face2AI their face with the 'Learn person' button if they want. "
                    "Do not guess a name."
                ),
                fallback_text=_fallback_unknown(lang),
            )
        if transition.to_state == "MULTIPLE_FACES":
            if not self.config.announce_multiple:
                return None
            if self.last_multiple_at is not None and (now - self.last_multiple_at).total_seconds() < MULTIPLE_ANNOUNCE_COOLDOWN_SECONDS:
                return None
            self.last_multiple_at = now
            return Prompt(
                kind="announce_multiple",
                instructions=(
                    f"Several people are now in front of the camera ({transition.faces}). "
                    f"In one short sentence in {lang}, note that you see more than one person and cannot tell who is who right now."
                ),
                fallback_text=_fallback_multiple(lang),
            )
        return None

    def on_store_change(self, change: StoreChange, now: datetime | None = None) -> Prompt | None:
        now = now or datetime.now(timezone.utc)
        lang = self.config.language
        if (now - change.at).total_seconds() > MAX_TRANSITION_AGE_SECONDS:
            return None
        if change.kind == "enrolled" and change.display_name:
            if change.identity_id:
                self.last_known[change.identity_id] = now
                self.present_identity = change.identity_id
            return Prompt(
                kind="welcome_enrolled",
                display_name=change.display_name,
                identity_id=change.identity_id,
                instructions=(
                    f"{change.display_name} was just enrolled in Face2AI (their face is now stored locally). "
                    f"Welcome them by name in one short sentence in {lang} and say you will recognize them from now on."
                ),
                fallback_text=_fallback_welcome(change.display_name, lang),
            )
        return None


def _fallback_known(name: str, lang: str) -> str:
    return f"Hi {name}." if lang.lower().startswith("en") else f"Hallo {name}."


def _fallback_unknown(lang: str) -> str:
    if lang.lower().startswith("en"):
        return "Hello! I don't know you yet — you can teach me your face with the Learn person button."
    return "Hallo! Ich kenne dich noch nicht – mit dem Knopf „Learn person“ kannst du mir dein Gesicht beibringen."


def _fallback_multiple(lang: str) -> str:
    if lang.lower().startswith("en"):
        return "I see more than one person, so I can't tell who is who right now."
    return "Ich sehe mehrere Personen, deshalb kann ich gerade nicht sagen, wer wer ist."


def _fallback_welcome(name: str, lang: str) -> str:
    if lang.lower().startswith("en"):
        return f"Welcome, {name}! I'll recognize you from now on."
    return f"Willkommen, {name}! Ab jetzt erkenne ich dich."


LANGUAGE_NAMES = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian"}


def build_instructions(config: AgentConfig, memory: PresenceMemory, now: datetime | None = None) -> str:
    """System prompt: persona + hard rules + the live presence report."""
    language = LANGUAGE_NAMES.get(config.language.lower(), config.language)
    persona = config.persona or (
        f"You are {config.agent_name}, a calm, warm, slightly playful voice companion running locally on this computer. "
        "You talk with the person in front of the camera like a good host: brief, natural, spoken language, no lists, no markdown."
    )
    rules = (
        f"Always answer in {language} unless the person clearly speaks another language. Keep replies to one or two spoken sentences.\n"
        "You can see who is present only through Face2AI's face recognition, summarized below and via your tools. "
        "Recognition is a best-effort match, never certainty and never a login: if someone says they are somebody else, believe them and say so kindly.\n"
        "Never claim to see, store or remember faces yourself; Face2AI stores only a name and a face encoding on this device, and people can delete themselves any time.\n"
        "Never invent a name. If the presence report says the person is unknown, do not guess who they are.\n"
        "Never read out identifiers, distances or technical details unless explicitly asked.\n"
        "Mood hints in the presence report ('wirkt fröhlich' / 'looks happy' …) are guesses from facial expression, never facts: "
        "never state them as facts, do not psychoanalyse, mention them at most in passing and with reservation "
        "('du wirkst …' / 'you look …'), never probe or press because of them, and never let them change how you greet or treat someone."
    )
    situation = memory.describe(now, language=config.language)
    return f"{persona}\n\nRules:\n{rules}\n\nCurrent situation (updated live):\n{situation}"
