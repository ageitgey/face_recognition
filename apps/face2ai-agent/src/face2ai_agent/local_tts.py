"""TTS over a local OpenAI-compatible ``/v1/audio/speech`` endpoint (speaches, Kokoro-FastAPI).

The stock ``livekit.plugins.openai.TTS`` speaks OpenAI's SSE streaming dialect for every model
outside {tts-1, tts-1-hd}; local servers return plain audio, so we do the one HTTP call ourselves
and hand the WAV/MP3 bytes to LiveKit's audio emitter, which decodes and resamples them.
"""

from __future__ import annotations

import uuid

import httpx
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError, tts

OUTPUT_SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


class LocalSpeechTTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        voice: str,
        api_key: str = "not-needed",
        response_format: str = "wav",
        speed: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._api_key = api_key
        self._response_format = response_format
        self._speed = speed
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "local-openai-compatible"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()) -> "LocalChunkedStream":
        return LocalChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._client.aclose()


class LocalChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: LocalSpeechTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._local = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "input": self.input_text,
            "model": self._local._model,
            "voice": self._local._voice,
            "response_format": self._local._response_format,
            "speed": self._local._speed,
        }
        headers = {"Authorization": f"Bearer {self._local._api_key}"}
        try:
            async with self._local._client.stream(
                "POST", f"{self._local._base_url}/audio/speech", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise APIStatusError(body[:300], status_code=response.status_code, request_id=None, body=body)
                output_emitter.initialize(
                    request_id=response.headers.get("x-request-id") or f"local-tts-{uuid.uuid4().hex[:12]}",
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._local._response_format}",
                )
                async for chunk in response.aiter_bytes():
                    if chunk:
                        output_emitter.push(chunk)
            output_emitter.flush()
        except httpx.TimeoutException as exc:
            raise APITimeoutError() from exc
        except APIStatusError:
            raise
        except httpx.HTTPError as exc:
            raise APIConnectionError() from exc
