"""Main module for MLX Audio API server.

This module provides a FastAPI-based server for hosting MLX Audio models,
including Text-to-Speech (TTS), Speech-to-Text (STT), and Speech-to-Speech (S2S) models.
It offers an OpenAI-compatible API for Audio completions and model management.
"""

import argparse
import asyncio
import base64
import inspect
import io
import json
import os
import re
import subprocess
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import mlx.core as mx
import numpy as np
import uvicorn
import webrtcvad
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import BaseModel

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from mlx_audio.realtime_vad import (
    VAD_SAMPLE_RATE,
    ServerVadConfig,
    StreamingVad,
    TurnDetectionError,
    TurnEventKind,
    parse_turn_detection,
)
from mlx_audio.server_inference import (
    BaseModelExecutionAdapter,
    InferenceBroker,
    InferenceHandle,
    InferenceRequest,
    InferenceResultChunk,
)
from mlx_audio.tts.continuous import TTSBatchItem, TTSBatchOptions
from mlx_audio.utils import load_model


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize NaN, Infinity, and -Infinity values for JSON serialization."""
    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return float(obj)
    else:
        return obj


class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    def load_model(self, model_name: str):
        if model_name not in self.models:
            self.models[model_name] = load_model(model_name)

        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                return True
            return False

    async def get_available_models(self):
        async with self.lock:
            return list(self.models.keys())


app = FastAPI()


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    """(Re)configure CORS middleware with the given origins."""
    # Remove any previously configured CORSMiddleware to avoid duplicates
    app.user_middleware = [
        m for m in app.user_middleware if m.cls is not CORSMiddleware
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Apply default CORS configuration when imported. The environment variable
# ``MLX_AUDIO_ALLOWED_ORIGINS`` can override the allowed origins by providing a
# comma-separated list. This ensures CORS headers are present even when running
# ``uvicorn mlx_audio.server:app`` directly.

allowed_origins_env = os.getenv("MLX_AUDIO_ALLOWED_ORIGINS")
default_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",")]
    if allowed_origins_env
    else ["*"]
)

# Setup CORS
setup_cors(app, default_origins)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    del app
    try:
        yield
    finally:
        global INFERENCE_BROKER
        if INFERENCE_BROKER is not None:
            INFERENCE_BROKER.stop_and_join()
            INFERENCE_BROKER = None


app.router.lifespan_context = app_lifespan


# Request schemas for OpenAI-compatible endpoints
class SpeechRequest(BaseModel):
    model: str
    input: str
    instruct: str | None = None
    voice: str | None = None
    speed: float | None = 1.0
    gender: str | None = "male"
    pitch: float | None = 1.0
    lang_code: str | None = "a"
    ref_audio: str | None = None
    ref_text: str | None = None
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    top_k: int | None = 40
    repetition_penalty: float | None = 1.0
    response_format: str | None = "mp3"
    stream: bool = False
    streaming_interval: float = 2.0
    max_tokens: int = 1200
    verbose: bool = False


class TranscriptionRequest(BaseModel):
    model: str
    language: str | None = None
    verbose: bool = False
    max_tokens: int = 1024
    chunk_duration: float = 30.0
    frame_threshold: int = 25
    stream: bool = False
    context: str | None = None
    prefill_step_size: int = 2048
    text: str | None = None
    word_timestamps: bool = False
    timestamp_granularities: Optional[str] = None


class SeparationResponse(BaseModel):
    target: str  # Base64 encoded WAV
    residual: str  # Base64 encoded WAV
    sample_rate: int


class ReFeliciaTTSRequest(BaseModel):
    text: str
    model: str | None = None
    voice: str | None = None
    lang_code: str | None = None
    ref_audio: str | None = None
    instruct: str | None = None
    caption: str | None = None
    speed: float | None = 1.0
    pitch_scale: float | None = 1.0
    intonation_scale: float | None = 1.0
    delivery_mode: str | None = "full_audio"
    streaming_interval: float | None = 0.24
    seconds: float | None = None
    duration_scale: float | None = None
    num_steps: int | None = None
    sequence_length: int | None = None
    cfg_guidance_mode: str | None = None
    t_schedule_mode: str | None = None
    sway_coeff: float | None = None
    segment_target_chars: int | None = 18
    segment_max_chars: int | None = 28
    crossfade_ms: int | None = 80
    trim_silence_ms: int | None = 60
    silence_threshold: int | None = 192


# Initialize the ModelProvider
model_provider = ModelProvider()
REALTIME_INFERENCE_LOCK = asyncio.Lock()
INFERENCE_BROKER: Optional[InferenceBroker] = None


@dataclass
class SpeechTaskPayload:
    request: SpeechRequest


@dataclass
class TranscriptionTaskPayload:
    request: TranscriptionRequest
    filename: str
    audio: np.ndarray
    sample_rate: int


@dataclass
class SeparationTaskPayload:
    model_name: str
    audio: np.ndarray
    sample_rate: int
    description: str
    method: str
    steps: int


def _load_model_for_inference(model_name: str):
    return model_provider.load_model(model_name)


async def _preflight_model_load(model_name: str) -> None:
    """Load ``model_name`` synchronously and translate failures into HTTPException.

    Routes that return a ``StreamingResponse`` commit the HTTP status + headers
    before the body generator runs, so any failure inside the inference worker
    surfaces as 200 OK with an empty body. Pre-flighting the load here lets the
    framework's exception handlers turn the failure into a real HTTP error
    response. Warm models are a no-op (``ModelProvider.load_model`` is cached).
    """
    try:
        await asyncio.to_thread(_load_model_for_inference, model_name)
    except HTTPException:
        raise
    except RepositoryNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name!r} is not a known HuggingFace repo or local path",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {model_name!r}: {exc}",
        ) from exc


_STT_EXTRA_KWARGS = {"word_timestamps", "timestamp_granularities"}


class STTExecutionAdapter(BaseModelExecutionAdapter):
    def run_serial(self, request: InferenceRequest) -> None:
        payload: TranscriptionTaskPayload = request.payload
        _, ext = os.path.splitext(payload.filename or "")
        suffix = ext or ".mp3"
        tmp_path = f"/tmp/{time.time()}_{request.request_id}{suffix}"
        audio_write(tmp_path, payload.audio, payload.sample_rate)

        try:
            stt_model = _load_model_for_inference(request.model_name)
            gen_kwargs = payload.request.model_dump(
                exclude={"model"}, exclude_none=True
            )
            signature = inspect.signature(stt_model.generate)
            gen_kwargs = {
                key: value
                for key, value in gen_kwargs.items()
                if key in signature.parameters or key in _STT_EXTRA_KWARGS
            }

            result = stt_model.generate(tmp_path, **gen_kwargs)
            if hasattr(result, "__iter__") and hasattr(result, "__next__"):
                accumulated_text = ""
                for chunk in result:
                    if request.cancel_event.is_set():
                        break
                    if isinstance(chunk, str):
                        accumulated_text += chunk
                        chunk_data = {
                            "text": chunk,
                            "accumulated": accumulated_text,
                        }
                    else:
                        chunk_data = {
                            "text": chunk.text,
                            "start": getattr(chunk, "start_time", None),
                            "end": getattr(chunk, "end_time", None),
                            "is_final": getattr(chunk, "is_final", None),
                            "language": getattr(chunk, "language", None),
                        }
                    request.emit_data(json.dumps(sanitize_for_json(chunk_data)) + "\n")
            elif not request.cancel_event.is_set():
                request.emit_data(json.dumps(sanitize_for_json(result)) + "\n")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        request.emit_done()


class _TTSAdapterContinuousSession:
    def __init__(
        self,
        *,
        adapter: "TTSExecutionAdapter",
        model_session,
    ):
        self.adapter = adapter
        self.model_session = model_session
        self._requests: dict[int, InferenceRequest] = {}
        self._emitted_audio: set[int] = set()
        self._next_sequence_id = 0

    @property
    def idle(self) -> bool:
        return self.model_session.idle and not self._requests

    def submit(self, request: InferenceRequest) -> None:
        if request.cancel_event.is_set():
            request.emit_done()
            return

        payload: SpeechTaskPayload = request.payload
        speech_request = payload.request
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self._requests[sequence_id] = request
        self.model_session.add(
            [
                TTSBatchItem(
                    sequence_id=sequence_id,
                    text=speech_request.input,
                    voice=speech_request.voice,
                    instruct=speech_request.instruct,
                    speed=speech_request.speed,
                    gender=speech_request.gender,
                    pitch=speech_request.pitch,
                    ref_audio=speech_request.ref_audio,
                    ref_text=speech_request.ref_text,
                )
            ]
        )

    def step(self) -> None:
        self._cancel_disconnected_requests()
        events = self.model_session.step()
        for event in events:
            request = self._requests.get(event.sequence_id)
            if request is None:
                continue

            if event.error is not None:
                request.emit_error(event.error)
                request.emit_done()
                self._requests.pop(event.sequence_id, None)
                self._emitted_audio.discard(event.sequence_id)
                continue

            if event.audio is not None and event.samples > 0:
                speech_request: SpeechRequest = request.payload.request
                if event.sample_rate is None:
                    request.emit_error(
                        ValueError("TTS batch event included audio without sample_rate")
                    )
                    request.emit_done()
                    self._requests.pop(event.sequence_id, None)
                    self._emitted_audio.discard(event.sequence_id)
                    continue
                self.adapter._emit_audio(
                    request,
                    speech_request,
                    event.audio,
                    event.sample_rate,
                )
                self._emitted_audio.add(event.sequence_id)

            if event.done:
                if event.sequence_id not in self._emitted_audio:
                    request.emit_error(
                        HTTPException(status_code=400, detail="No audio generated")
                    )
                request.emit_done()
                self._requests.pop(event.sequence_id, None)
                self._emitted_audio.discard(event.sequence_id)

    def fail(self, error: BaseException) -> None:
        for request in list(self._requests.values()):
            request.emit_error(error)
            request.emit_done()
        self._requests.clear()
        self._emitted_audio.clear()

    def _cancel_disconnected_requests(self) -> None:
        for sequence_id, request in list(self._requests.items()):
            if not request.cancel_event.is_set():
                continue
            self.model_session.cancel(sequence_id)
            request.emit_done()
            self._requests.pop(sequence_id, None)
            self._emitted_audio.discard(sequence_id)


class TTSExecutionAdapter(BaseModelExecutionAdapter):
    _REQUEST_MODEL_ATTR = "_mlx_audio_loaded_model"

    def __init__(self):
        raw_max_batch_size = os.getenv("MLX_AUDIO_TTS_MAX_BATCH_SIZE", "8")
        try:
            self.max_batch_size = max(1, int(raw_max_batch_size))
        except ValueError:
            self.max_batch_size = 8

    def _get_model_for_request(self, request: InferenceRequest):
        model = getattr(request, self._REQUEST_MODEL_ATTR, None)
        if model is None:
            model = _load_model_for_inference(request.model_name)
            setattr(request, self._REQUEST_MODEL_ATTR, model)
        return model

    def _build_batch_options(self, request: InferenceRequest) -> TTSBatchOptions:
        speech_request: SpeechRequest = request.payload.request
        return TTSBatchOptions(
            temperature=speech_request.temperature,
            top_p=speech_request.top_p,
            top_k=speech_request.top_k,
            repetition_penalty=speech_request.repetition_penalty,
            max_tokens=speech_request.max_tokens,
            lang_code=speech_request.lang_code,
            stream=speech_request.stream,
            streaming_interval=speech_request.streaming_interval,
            max_batch_size=self.max_batch_size,
            verbose=speech_request.verbose,
        )

    def _request_kwargs(self, request: InferenceRequest) -> dict[str, Any]:
        speech_request: SpeechRequest = request.payload.request
        return speech_request.model_dump(exclude={"model"}, exclude_none=True)

    def _get_callable_model_attr(self, model, name: str):
        value = getattr(model, name, None)
        if not callable(value):
            return None

        # MagicMock fabricates missing attributes on demand; only treat explicit
        # mock attributes as model hooks in tests.
        if type(model).__module__ == "unittest.mock" and name not in vars(model):
            return None
        return value

    def _model_supports_request(
        self,
        model,
        request: InferenceRequest,
        *,
        continuous: bool,
    ) -> bool:
        speech_request: SpeechRequest = request.payload.request
        hook_name = (
            "supports_tts_continuous_batch" if continuous else "supports_tts_batch"
        )
        hook = self._get_callable_model_attr(model, hook_name)
        if callable(hook):
            return bool(hook(**self._request_kwargs(request)))

        if speech_request.stream:
            return False
        if speech_request.ref_audio or speech_request.ref_text:
            return False
        if speech_request.gender not in (None, "male"):
            return False
        if speech_request.speed not in (None, 1.0):
            return False
        if speech_request.pitch not in (None, 1.0):
            return False
        return True

    def _can_call_batch_generate(self, model, request: InferenceRequest) -> bool:
        batch_generate = self._get_callable_model_attr(model, "batch_generate")
        if batch_generate is None:
            return False
        if not self._model_supports_request(model, request, continuous=False):
            return False

        speech_request: SpeechRequest = request.payload.request
        if speech_request.stream:
            return False

        signature = inspect.signature(batch_generate)
        params = signature.parameters
        if "texts" not in params:
            return False
        if speech_request.voice and "voices" not in params:
            return False
        if speech_request.instruct and "instructs" not in params:
            return False
        if speech_request.gender not in (None, "male") and "genders" not in params:
            return False
        if speech_request.speed not in (None, 1.0) and "speeds" not in params:
            return False
        if speech_request.pitch not in (None, 1.0) and "pitches" not in params:
            return False
        if speech_request.ref_audio and "ref_audios" not in params:
            return False
        if speech_request.ref_text and "ref_texts" not in params:
            return False
        return True

    def _can_call_continuous_session(self, model, request: InferenceRequest) -> bool:
        create_session = self._get_callable_model_attr(
            model,
            "create_tts_batch_session",
        )
        return create_session is not None and self._model_supports_request(
            model, request, continuous=True
        )

    def supports_batch(self, request: InferenceRequest) -> bool:
        model = self._get_model_for_request(request)
        return self._can_call_batch_generate(model, request)

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        model = self._get_model_for_request(request)
        return self._can_call_continuous_session(model, request)

    def batch_key(self, request: InferenceRequest) -> Any:
        speech_request: SpeechRequest = request.payload.request
        return (
            "tts",
            speech_request.stream,
            speech_request.lang_code,
            speech_request.temperature,
            speech_request.top_p,
            speech_request.top_k,
            speech_request.repetition_penalty,
            speech_request.max_tokens,
            speech_request.ref_audio,
            speech_request.ref_text,
            speech_request.streaming_interval if speech_request.stream else None,
            speech_request.verbose,
        )

    def continuous_batch_key(self, request: InferenceRequest) -> Any:
        return self.batch_key(request)

    def create_continuous_batch_session(self, request: InferenceRequest):
        model = self._get_model_for_request(request)
        create_session = self._get_callable_model_attr(
            model,
            "create_tts_batch_session",
        )
        if create_session is None:
            raise NotImplementedError("Model does not provide create_tts_batch_session")
        model_session = create_session(self._build_batch_options(request))
        return _TTSAdapterContinuousSession(
            adapter=self,
            model_session=model_session,
        )

    def _emit_audio(
        self,
        request: InferenceRequest,
        speech_request: SpeechRequest,
        audio,
        sample_rate: int,
    ) -> None:
        buffer = io.BytesIO()
        audio_write(
            buffer,
            audio,
            sample_rate,
            format=speech_request.response_format,
        )
        request.emit_data(buffer.getvalue())

    def run_serial(self, request: InferenceRequest) -> None:
        payload: SpeechTaskPayload = request.payload
        speech_request = payload.request
        model = self._get_model_for_request(request)

        ref_audio = speech_request.ref_audio
        if ref_audio and isinstance(ref_audio, str):
            if not os.path.exists(ref_audio):
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference audio file not found: {ref_audio}",
                )

            from mlx_audio.tts.generate import load_audio

            normalize = hasattr(model, "model_type") and model.model_type == "spark"
            ref_audio = load_audio(
                ref_audio,
                sample_rate=model.sample_rate,
                volume_normalize=normalize,
            )

        audio_chunks = []
        sample_rate = None
        generate_kwargs = {
            "voice": speech_request.voice,
            "speed": speech_request.speed,
            "gender": speech_request.gender,
            "pitch": speech_request.pitch,
            "instruct": speech_request.instruct,
            "lang_code": speech_request.lang_code,
            "ref_audio": ref_audio,
            "ref_text": speech_request.ref_text,
            "temperature": speech_request.temperature,
            "top_p": speech_request.top_p,
            "top_k": speech_request.top_k,
            "repetition_penalty": speech_request.repetition_penalty,
            "stream": speech_request.stream,
            "streaming_interval": speech_request.streaming_interval,
            "max_tokens": speech_request.max_tokens,
            "verbose": speech_request.verbose,
        }
        if (
            _is_kokoro_tts_model(model, request.model_name)
            and _contains_japanese_text(speech_request.input)
            and speech_request.lang_code in (None, "a")
        ):
            generate_kwargs["lang_code"] = "j"
            generate_kwargs["voice"] = _kokoro_voice_for_lang(
                generate_kwargs.get("voice"),
                "j",
            )

        for result in model.generate(speech_request.input, **generate_kwargs):
            if request.cancel_event.is_set():
                mx.clear_cache()
                request.emit_done()
                return

            if speech_request.stream:
                self._emit_audio(
                    request,
                    speech_request,
                    result.audio,
                    result.sample_rate,
                )
            else:
                audio_chunks.append(result.audio)
                if sample_rate is None:
                    sample_rate = result.sample_rate

        if speech_request.stream:
            request.emit_done()
            return

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio generated")

        concatenated_audio = np.concatenate(audio_chunks)
        self._emit_audio(request, speech_request, concatenated_audio, sample_rate)
        request.emit_done()

    def run_batch(self, requests: list[InferenceRequest]) -> None:
        if len(requests) == 1:
            self.run_serial(requests[0])
            return

        model = self._get_model_for_request(requests[0])
        if not all(
            self._can_call_batch_generate(model, request) for request in requests
        ):
            for request in requests:
                self.run_serial(request)
            return

        first_speech_request: SpeechRequest = requests[0].payload.request
        texts = [request.payload.request.input for request in requests]
        voices = [request.payload.request.voice for request in requests]
        instructs = [request.payload.request.instruct for request in requests]
        speeds = [request.payload.request.speed for request in requests]
        genders = [request.payload.request.gender for request in requests]
        pitches = [request.payload.request.pitch for request in requests]
        ref_audios = [request.payload.request.ref_audio for request in requests]
        ref_texts = [request.payload.request.ref_text for request in requests]

        batch_generate = self._get_callable_model_attr(model, "batch_generate")
        if batch_generate is None:
            for request in requests:
                self.run_serial(request)
            return

        signature = inspect.signature(batch_generate)
        kwargs = {
            "texts": texts,
            "voices": voices,
            "instructs": instructs,
            "speeds": speeds,
            "genders": genders,
            "pitches": pitches,
            "ref_audios": ref_audios,
            "ref_texts": ref_texts,
            "temperature": first_speech_request.temperature,
            "lang_code": first_speech_request.lang_code,
            "max_tokens": first_speech_request.max_tokens,
            "top_k": first_speech_request.top_k,
            "top_p": first_speech_request.top_p,
            "repetition_penalty": first_speech_request.repetition_penalty,
            "stream": False,
            "verbose": first_speech_request.verbose,
        }
        kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }

        audio_chunks_by_sequence: list[list[Any]] = [[] for _ in requests]
        sample_rates: list[int | None] = [None for _ in requests]

        for result in batch_generate(**kwargs):
            sequence_idx = result.sequence_idx
            if sequence_idx < 0 or sequence_idx >= len(requests):
                continue
            request = requests[sequence_idx]
            if request.cancel_event.is_set():
                continue
            audio_chunks_by_sequence[sequence_idx].append(result.audio)
            sample_rates[sequence_idx] = result.sample_rate

        for sequence_idx, request in enumerate(requests):
            if request.cancel_event.is_set():
                request.emit_done()
                continue

            chunks = audio_chunks_by_sequence[sequence_idx]
            sample_rate = sample_rates[sequence_idx]
            if not chunks or sample_rate is None:
                request.emit_error(
                    HTTPException(status_code=400, detail="No audio generated")
                )
                request.emit_done()
                continue

            audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            self._emit_audio(
                request,
                request.payload.request,
                audio,
                sample_rate,
            )
            request.emit_done()


class SeparationExecutionAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self._model_name: str | None = None
        self._model = None
        self._processor = None

    def _get_resources(self, model_name: str):
        if self._model_name == model_name and self._model is not None:
            return self._model, self._processor

        from mlx_audio.sts import SAMAudio, SAMAudioProcessor

        self._processor = SAMAudioProcessor.from_pretrained(model_name)
        self._model = SAMAudio.from_pretrained(model_name)
        self._model_name = model_name
        return self._model, self._processor

    def run_serial(self, request: InferenceRequest) -> None:
        payload: SeparationTaskPayload = request.payload
        tmp_path = f"/tmp/separation_{time.time()}_{request.request_id}.wav"
        audio_write(tmp_path, payload.audio, payload.sample_rate)

        try:
            model, processor = self._get_resources(payload.model_name)
            batch = processor(
                descriptions=[payload.description],
                audios=[tmp_path],
            )

            step_size = 2 / (payload.steps * 2)
            ode_opt = {"method": payload.method, "step_size": step_size}

            result = model.separate_long(
                audios=batch.audios,
                descriptions=batch.descriptions,
                anchor_ids=batch.anchor_ids,
                anchor_alignment=batch.anchor_alignment,
                ode_opt=ode_opt,
                ode_decode_chunk_size=50,
            )

            mx.clear_cache()

            target_audio = np.array(result.target[0])
            residual_audio = np.array(result.residual[0])
            sample_rate = model.sample_rate

            def audio_to_base64(audio_array, sr):
                buffer = io.BytesIO()
                audio_write(buffer, audio_array, sr, format="wav")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")

            request.emit_data(
                SeparationResponse(
                    target=audio_to_base64(target_audio, sample_rate),
                    residual=audio_to_base64(residual_audio, sample_rate),
                    sample_rate=sample_rate,
                )
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        request.emit_done()


def get_inference_broker() -> InferenceBroker:
    global INFERENCE_BROKER
    if INFERENCE_BROKER is None:
        broker = InferenceBroker()
        broker.register_adapter("stt", STTExecutionAdapter())
        broker.register_adapter("tts", TTSExecutionAdapter())
        broker.register_adapter("separation", SeparationExecutionAdapter())
        INFERENCE_BROKER = broker
    return INFERENCE_BROKER


async def _next_inference_chunk(handle: InferenceHandle) -> InferenceResultChunk:
    return await asyncio.to_thread(handle.result_queue.get)


async def _stream_inference_results(handle: InferenceHandle, request: Request):
    try:
        while True:
            chunk = await _next_inference_chunk(handle)
            if chunk.kind == "done":
                break
            if chunk.kind == "error":
                raise chunk.error
            yield chunk.payload
            await asyncio.sleep(0)
            if await request.is_disconnected():
                handle.cancel()
                break
    finally:
        handle.cancel()


async def _await_inference_result(handle: InferenceHandle):
    result = None
    try:
        while True:
            chunk = await _next_inference_chunk(handle)
            if chunk.kind == "done":
                return result
            if chunk.kind == "error":
                raise chunk.error
            result = chunk.payload
    finally:
        handle.cancel()


def _resolve_refelicia_tts_model(model_name: str | None) -> str:
    configured_model = os.getenv("MLX_AUDIO_REFELICIA_TTS_MODEL")
    if configured_model and (not model_name or model_name == "amitaro"):
        return configured_model
    if model_name:
        return model_name
    if configured_model:
        return configured_model
    raise HTTPException(
        status_code=400,
        detail=(
            "TTS model is required. Send a model in the WebSocket request or set "
            "MLX_AUDIO_REFELICIA_TTS_MODEL."
        ),
    )


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _is_irodori_tts_model(model: Any, model_name: str) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    return model_type == "irodori_tts" or "irodori" in model_name.lower()


def _is_kokoro_tts_model(model: Any, model_name: str) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    return model_type == "kokoro" or "kokoro" in model_name.lower()


def _contains_japanese_text(text: str | None) -> bool:
    if not text:
        return False
    return any(
        ("\u3040" <= char <= "\u30ff")
        or ("\u3400" <= char <= "\u9fff")
        for char in text
    )


def _kokoro_voice_for_lang(voice: str | None, lang_code: str | None) -> str | None:
    if lang_code != "j":
        return voice
    if not voice or voice.startswith(("a", "b", "e", "f", "h", "i", "p", "z")):
        return os.getenv("MLX_AUDIO_REFELICIA_TTS_VOICE") or "jf_alpha"
    return voice


def _refelicia_lang_code(
    payload: ReFeliciaTTSRequest,
    *,
    model: Any,
    model_name: str,
) -> str | None:
    lang_code = payload.lang_code or os.getenv("MLX_AUDIO_REFELICIA_TTS_LANG_CODE")
    if lang_code:
        return lang_code

    if not _is_kokoro_tts_model(model, model_name):
        return None

    if _contains_japanese_text(payload.text):
        return "j"

    voice = (payload.voice or os.getenv("MLX_AUDIO_REFELICIA_TTS_VOICE") or "").strip()
    if len(voice) >= 2 and voice[0] in "abefhipjz" and voice[1] in "fm":
        return voice[0]
    return None


_REFELICIA_REF_AUDIO_CACHE: dict[tuple[str, float, int], Any] = {}


def _load_refelicia_ref_audio(path: str, sample_rate: int) -> Any:
    """Load and resample reference audio once, then reuse across requests.

    Keyed by (path, mtime, sample_rate) so replacing the file on disk
    invalidates the cached waveform.
    """
    key = (os.path.abspath(path), os.path.getmtime(path), sample_rate)
    cached = _REFELICIA_REF_AUDIO_CACHE.get(key)
    if cached is None:
        from mlx_audio.tts.generate import load_audio

        cached = load_audio(path, sample_rate=sample_rate)
        mx.eval(cached)
        _REFELICIA_REF_AUDIO_CACHE.clear()
        _REFELICIA_REF_AUDIO_CACHE[key] = cached
    return cached


def _refelicia_irodori_kwargs(
    payload: ReFeliciaTTSRequest,
    *,
    model: Any,
    model_name: str,
) -> dict[str, Any]:
    if not _is_irodori_tts_model(model, model_name):
        return {}

    ref_audio = payload.ref_audio or os.getenv("MLX_AUDIO_REFELICIA_TTS_REF_AUDIO")
    caption = (
        payload.caption
        or payload.instruct
        or os.getenv("MLX_AUDIO_REFELICIA_TTS_CAPTION")
        or os.getenv("MLX_AUDIO_REFELICIA_TTS_INSTRUCT")
    )

    voice = (payload.voice or "").strip()
    if voice and not ref_audio and os.path.exists(voice):
        ref_audio = voice
    elif voice and not caption:
        caption = voice

    if ref_audio and not os.path.exists(ref_audio):
        raise HTTPException(
            status_code=400,
            detail=f"Reference audio file not found: {ref_audio}",
        )
    if ref_audio:
        ref_audio = _load_refelicia_ref_audio(
            ref_audio,
            int(getattr(model, "sample_rate", 24000) or 24000),
        )

    kwargs: dict[str, Any] = {
        "ref_audio": ref_audio,
        "instruct": caption,
        "seconds": payload.seconds or _env_float("MLX_AUDIO_REFELICIA_TTS_SECONDS"),
        "duration_scale": payload.duration_scale
        or _env_float("MLX_AUDIO_REFELICIA_TTS_DURATION_SCALE"),
        "num_steps": payload.num_steps
        or _env_int("MLX_AUDIO_REFELICIA_TTS_NUM_STEPS"),
        "sequence_length": payload.sequence_length
        or _env_int("MLX_AUDIO_REFELICIA_TTS_SEQUENCE_LENGTH"),
        "cfg_guidance_mode": payload.cfg_guidance_mode
        or os.getenv("MLX_AUDIO_REFELICIA_TTS_CFG_GUIDANCE_MODE"),
        "t_schedule_mode": payload.t_schedule_mode
        or os.getenv("MLX_AUDIO_REFELICIA_TTS_T_SCHEDULE_MODE"),
        "sway_coeff": payload.sway_coeff
        or _env_float("MLX_AUDIO_REFELICIA_TTS_SWAY_COEFF"),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


# Segmented streaming for the ReFelicia TTS WebSocket. Mirrors the semantics of
# ReFeliciaPython/style_bert_vits2_openai_server.py: a short first segment for
# low TTFA, larger follow-up segments, silence trimming at the seams, and a
# short crossfade between consecutive segments.
_REFELICIA_SEGMENT_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?\n])")
_REFELICIA_CLAUSE_SPLIT_PATTERN = re.compile(r"(?<=[、,])")
_REFELICIA_MIN_FIRST_SEGMENT_CHARS = 8


def _split_long_refelicia_segment(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = [
        part.strip()
        for part in _REFELICIA_CLAUSE_SPLIT_PATTERN.split(text)
        if part.strip()
    ]
    merged: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > max_chars:
            merged.append(current)
            current = part
        else:
            current += part
    if current:
        merged.append(current)
    segments: list[str] = []
    for segment in merged:
        segments.extend(
            segment[index : index + max_chars]
            for index in range(0, len(segment), max_chars)
        )
    return segments


def _merge_refelicia_clauses(
    clauses: list[str],
    *,
    target_chars: int,
    max_chars: int,
) -> list[str]:
    segments: list[str] = []
    current = ""
    for clause in clauses:
        candidate = current + clause if current else clause
        if current and len(candidate) > max_chars:
            segments.extend(_split_long_refelicia_segment(current, max_chars))
            current = clause
        elif len(candidate) >= target_chars:
            segments.extend(_split_long_refelicia_segment(candidate, max_chars))
            current = ""
        else:
            current = candidate
    if current:
        segments.extend(_split_long_refelicia_segment(current, max_chars))
    return segments


def _split_refelicia_text_segments(
    text: str,
    target_chars: int,
    max_chars: int,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    clauses = [
        clause.strip()
        for clause in _REFELICIA_SEGMENT_SPLIT_PATTERN.split(normalized)
        if clause.strip()
    ]
    if not clauses:
        return [normalized]
    while len(clauses) >= 2 and len(clauses[0]) < _REFELICIA_MIN_FIRST_SEGMENT_CHARS:
        clauses = [clauses[0] + clauses[1], *clauses[2:]]

    first_target_chars = max(6, min(target_chars, 10))
    first_max_chars = max(first_target_chars, min(max_chars, 14))
    follow_target_chars = max(max_chars, target_chars * 2)
    follow_max_chars = max(follow_target_chars, max_chars * 2)

    segments = _merge_refelicia_clauses(
        clauses[:1],
        target_chars=first_target_chars,
        max_chars=first_max_chars,
    )
    if clauses[1:]:
        segments.extend(
            _merge_refelicia_clauses(
                clauses[1:],
                target_chars=follow_target_chars,
                max_chars=follow_max_chars,
            )
        )
    return segments or [normalized]


def _trim_refelicia_pcm16_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_trim_ms: int,
    threshold: int,
    trim_start: bool,
    trim_end: bool,
) -> np.ndarray:
    if len(audio) == 0 or max_trim_ms <= 0 or threshold <= 0:
        return audio

    max_trim_samples = min(len(audio), int(sample_rate * max_trim_ms / 1000))
    start_index = 0
    end_index = len(audio)

    if trim_start and max_trim_samples > 0:
        while start_index < max_trim_samples and abs(int(audio[start_index])) <= threshold:
            start_index += 1

    if trim_end and max_trim_samples > 0:
        tail_limit = len(audio) - max_trim_samples
        while (
            end_index > tail_limit
            and end_index > start_index
            and abs(int(audio[end_index - 1])) <= threshold
        ):
            end_index -= 1

    trimmed = audio[start_index:end_index]
    return trimmed if len(trimmed) > 0 else audio


def _crossfade_refelicia_pcm16(
    left_tail: np.ndarray,
    right_head: np.ndarray,
    overlap_samples: int,
) -> np.ndarray:
    overlap = min(len(left_tail), len(right_head), overlap_samples)
    if overlap <= 0:
        return np.concatenate([left_tail, right_head]).astype(np.int16, copy=False)

    left = left_tail[-overlap:].astype(np.float32)
    right = right_head[:overlap].astype(np.float32)
    ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    fused = np.round(left * (1.0 - ramp) + right * ramp).astype(np.int16)
    return np.concatenate([fused, right_head[overlap:]]).astype(np.int16, copy=False)


def _audio_to_pcm16le(audio: Any) -> bytes:
    audio_array = np.asarray(audio)
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=-1)
    audio_array = np.nan_to_num(audio_array.astype(np.float32), copy=False)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    return (audio_array * 32767.0).astype("<i2").tobytes()


def _filter_generate_kwargs(generate, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(generate)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


@app.get("/")
async def root():
    return {
        "message": "Welcome to the MLX Audio API server! Visit https://localhost:3000 for the UI."
    }


@app.get("/v1/models")
async def list_models():
    """
    Get list of models - provided in OpenAI API compliant format.
    """
    models = await model_provider.get_available_models()
    models_data = []
    for model in models:
        models_data.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system",
            }
        )
    return {"object": "list", "data": models_data}


@app.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.

    Args:
        model_name (str): The name of the model to add.

    Returns:
        dict (dict): A dictionary containing the status of the operation.
    """
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
    """
    Remove a model from the API.

    Args:
        model_name (str): The name of the model to remove.

    Returns:
        Response (str): A 204 No Content response if successful.

    Raises:
        HTTPException (str): If the model is not found.
    """
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest, request: Request):
    """Generate speech audio following the OpenAI text-to-speech API."""
    if payload.ref_audio and isinstance(payload.ref_audio, str):
        if not os.path.exists(payload.ref_audio):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {payload.ref_audio}",
            )

    await _preflight_model_load(payload.model)

    handle = get_inference_broker().submit(
        endpoint_kind="tts",
        model_name=payload.model,
        payload=SpeechTaskPayload(request=payload),
        normalized_kwargs=payload.model_dump(exclude={"model"}, exclude_none=True),
        stream=payload.stream,
    )
    return StreamingResponse(
        _stream_inference_results(handle, request),
        media_type=f"audio/{payload.response_format}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        },
    )


@app.get("/v1/audio/voices")
async def tts_voices(model: Optional[str] = None):
    """List available voices for a TTS model.

    Resolves the model's HF snapshot directory and enumerates
    ``voices/*.safetensors`` files (the convention used by Kokoro and
    similar voice-pack-based TTS models). Returns an empty ``data`` list
    for models that don't ship per-voice packs so callers can fall back
    to whatever defaults make sense for that model.
    """
    if not model:
        raise HTTPException(status_code=400, detail="model query parameter is required")

    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model, allow_patterns=["voices/*.safetensors"]
        )
    except Exception as e:
        return {"object": "list", "model": model, "data": [], "error": str(e)}

    voices_dir = Path(snapshot) / "voices"
    if not voices_dir.is_dir():
        return {"object": "list", "model": model, "data": []}

    voices = sorted(p.stem for p in voices_dir.glob("*.safetensors"))
    return {
        "object": "list",
        "model": model,
        "data": [{"id": v, "name": v} for v in voices],
    }


@app.post("/v1/audio/transcriptions")
async def stt_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    verbose: bool = Form(False),
    max_tokens: int = Form(1024),
    chunk_duration: float = Form(30.0),
    frame_threshold: int = Form(25),
    stream: bool = Form(False),
    context: Optional[str] = Form(None),
    prefill_step_size: int = Form(2048),
    text: Optional[str] = Form(None),
    response_format: str = Form("ndjson"),
    word_timestamps: bool = Form(False),
    timestamp_granularities: Optional[str] = Form(None),
):
    """Transcribe audio using an STT model.

    The default ``response_format`` (``ndjson``) preserves mlx-audio's native
    ``application/x-ndjson`` streaming transport, where each line is a JSON
    object emitted by the underlying STT model (text deltas while streaming,
    or the full whisper segment payload for batch transcription).

    For OpenAI Audio API compatibility, ``response_format`` also accepts:

    * ``text`` -- ``text/plain`` body with the final transcript only.
    * ``json`` -- ``application/json`` body shaped ``{"text": "..."}``.
    * ``verbose_json`` -- ``application/json`` body with the full payload
      from the underlying model (``text``, ``segments``, ``language``,
      ...), passed through unchanged.

    See https://platform.openai.com/docs/api-reference/audio/createTranscription
    """
    payload = TranscriptionRequest(
        model=model,
        language=language,
        verbose=verbose,
        max_tokens=max_tokens,
        chunk_duration=chunk_duration,
        frame_threshold=frame_threshold,
        stream=stream,
        context=context,
        prefill_step_size=prefill_step_size,
        text=text,
        word_timestamps=word_timestamps,
        timestamp_granularities=timestamp_granularities,
    )
    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()

    await _preflight_model_load(payload.model)

    handle = get_inference_broker().submit(
        endpoint_kind="stt",
        model_name=payload.model,
        payload=TranscriptionTaskPayload(
            request=payload,
            filename=file.filename or "audio.mp3",
            audio=audio,
            sample_rate=sr,
        ),
        normalized_kwargs=payload.model_dump(exclude={"model"}, exclude_none=True),
        stream=payload.stream,
    )

    if response_format in ("text", "json", "verbose_json"):
        full: Optional[dict] = None
        accumulated = ""
        try:
            while True:
                chunk = await _next_inference_chunk(handle)
                if chunk.kind == "done":
                    break
                if chunk.kind == "error":
                    raise chunk.error
                line = chunk.payload
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                for raw in str(line).splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and (
                        "segments" in obj or "language" in obj
                    ):
                        full = obj
                    elif isinstance(obj, dict) and "text" in obj:
                        accumulated += obj.get("text") or ""
        finally:
            handle.cancel()

        if full is None:
            full = {"text": accumulated}

        if response_format == "text":
            return PlainTextResponse((full.get("text") or "").strip())
        if response_format == "json":
            return JSONResponse({"text": (full.get("text") or "").strip()})
        # verbose_json: full payload (text, segments, language, ...) as-is
        return JSONResponse(full)

    return StreamingResponse(
        _stream_inference_results(handle, request),
        media_type="application/x-ndjson",
    )


@app.post("/v1/audio/separations")
async def audio_separations(
    file: UploadFile = File(...),
    model: str = Form("mlx-community/sam-audio-large-fp16"),
    description: str = Form("speech"),
    method: str = Form("midpoint"),
    steps: int = Form(16),
):
    """Separate audio using SAM Audio model.

    Args:
        file: Audio file to process
        model: SAM Audio model name (default: mlx-community/sam-audio-large-fp16)
        description: Text description of what to separate (e.g., "speech", "guitar", "drums")
        method: ODE solver method - "midpoint" or "euler" (default: midpoint)
        steps: Number of ODE steps - 2, 4, 8, 16, or 32 (default: 16)

    Returns:
        JSON with base64-encoded target and residual audio, plus sample rate
    """
    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()

    handle = get_inference_broker().submit(
        endpoint_kind="separation",
        model_name=model,
        payload=SeparationTaskPayload(
            model_name=model,
            audio=audio,
            sample_rate=sr,
            description=description,
            method=method,
            steps=steps,
        ),
        normalized_kwargs={
            "description": description,
            "method": method,
            "steps": steps,
        },
    )
    return await _await_inference_result(handle)


async def _handle_refelicia_tts_request(
    websocket: WebSocket,
    payload: ReFeliciaTTSRequest,
) -> None:
    model_name = _resolve_refelicia_tts_model(payload.model)
    model = await asyncio.to_thread(_load_model_for_inference, model_name)

    generate_kwargs = {
        "voice": payload.voice or os.getenv("MLX_AUDIO_REFELICIA_TTS_VOICE") or None,
        "speed": payload.speed,
        "pitch": payload.pitch_scale,
        "stream": False,
        "streaming_interval": payload.streaming_interval,
        "verbose": False,
    }
    lang_code = _refelicia_lang_code(payload, model=model, model_name=model_name)
    if lang_code is not None:
        generate_kwargs["lang_code"] = lang_code
        if _is_kokoro_tts_model(model, model_name):
            generate_kwargs["voice"] = _kokoro_voice_for_lang(
                generate_kwargs.get("voice"),
                lang_code,
            )
    generate_kwargs.update(
        _refelicia_irodori_kwargs(payload, model=model, model_name=model_name)
    )
    generate_kwargs = _filter_generate_kwargs(model.generate, generate_kwargs)

    started_at = time.time()
    sample_rate = int(getattr(model, "sample_rate", 24000) or 24000)

    delivery_mode = (payload.delivery_mode or "full_audio").strip().lower()
    if delivery_mode != "full_audio":
        await _stream_refelicia_tts_segments(
            websocket,
            model=model,
            payload=payload,
            generate_kwargs=generate_kwargs,
            sample_rate=sample_rate,
            started_at=started_at,
        )
        return

    generated_chunks: list[tuple[bytes, int]] = []

    for result in model.generate(payload.text, **generate_kwargs):
        result_audio = getattr(result, "audio", None)
        if result_audio is None:
            continue
        result_sample_rate = int(getattr(result, "sample_rate", sample_rate))
        if result_sample_rate > 0:
            sample_rate = result_sample_rate

        pcm = _audio_to_pcm16le(result_audio)
        if pcm:
            generated_chunks.append((pcm, sample_rate))

    first_pcm_ready_seconds = time.time() - started_at if generated_chunks else 0.0

    dump_path = os.getenv("MLX_AUDIO_REFELICIA_TTS_DUMP_WAV")
    if dump_path and generated_chunks:
        dump_pcm = b"".join(chunk for chunk, _ in generated_chunks)
        dump_audio = np.frombuffer(dump_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        audio_write(dump_path, dump_audio, generated_chunks[-1][1])

    await websocket.send_json(
        {
            "type": "start",
            "mode": "tts",
            "delivery_mode": payload.delivery_mode or "full_audio",
            "sample_rate": sample_rate,
            "channels": 1,
            "format": "pcm_s16le",
        }
    )
    start_sent_seconds = time.time() - started_at

    chunks_sent = 0
    samples_sent = 0
    first_chunk_sent_seconds = 0.0
    for pcm, chunk_sample_rate in generated_chunks:
        sample_rate = chunk_sample_rate
        await websocket.send_bytes(pcm)
        chunks_sent += 1
        samples_sent += len(pcm) // 2
        if chunks_sent == 1:
            first_chunk_sent_seconds = time.time() - started_at

    generate_seconds = time.time() - started_at
    audio_seconds = samples_sent / sample_rate if sample_rate > 0 else 0.0
    await websocket.send_json(
        {
            "type": "end",
            "chunks": chunks_sent,
            "delivery_mode": payload.delivery_mode or "full_audio",
            "audio_seconds": audio_seconds,
            "generate_seconds": generate_seconds,
            "real_time_factor": generate_seconds / audio_seconds
            if audio_seconds > 0
            else 0.0,
            "first_pcm_ready_seconds": first_pcm_ready_seconds,
            "start_sent_seconds": start_sent_seconds,
            "first_chunk_sent_seconds": first_chunk_sent_seconds,
            "segments": 1 if chunks_sent else 0,
        }
    )
    _log_refelicia_tts_request(
        mode="full_audio",
        text=payload.text,
        segments=1,
        first_pcm_ready_seconds=first_pcm_ready_seconds,
        generate_seconds=generate_seconds,
        audio_seconds=audio_seconds,
    )


def _log_refelicia_tts_request(
    *,
    mode: str,
    text: str,
    segments: int,
    first_pcm_ready_seconds: float,
    generate_seconds: float,
    audio_seconds: float,
) -> None:
    rtf = generate_seconds / audio_seconds if audio_seconds > 0 else 0.0
    print(
        f"[refelicia-tts] mode={mode} chars={len(text)} segments={segments} "
        f"first_pcm={first_pcm_ready_seconds * 1000:.0f}ms "
        f"total={generate_seconds * 1000:.0f}ms "
        f"audio={audio_seconds:.2f}s rtf={rtf:.2f}",
        flush=True,
    )


async def _stream_refelicia_tts_segments(
    websocket: WebSocket,
    *,
    model: Any,
    payload: ReFeliciaTTSRequest,
    generate_kwargs: dict[str, Any],
    sample_rate: int,
    started_at: float,
) -> None:
    """Generate per text segment and send PCM as soon as each segment is ready.

    TTFA is bounded by the first (short) segment instead of the full utterance.
    """
    segments = _split_refelicia_text_segments(
        payload.text,
        payload.segment_target_chars or 18,
        payload.segment_max_chars or 28,
    )

    # Load the reference audio once so each segment does not re-read and
    # resample the file.
    ref_audio = generate_kwargs.get("ref_audio")
    if isinstance(ref_audio, str):
        from mlx_audio.tts.generate import load_audio

        generate_kwargs["ref_audio"] = load_audio(ref_audio, sample_rate=sample_rate)

    await websocket.send_json(
        {
            "type": "start",
            "mode": "tts",
            "delivery_mode": payload.delivery_mode,
            "sample_rate": sample_rate,
            "channels": 1,
            "format": "pcm_s16le",
        }
    )
    start_sent_seconds = time.time() - started_at

    overlap_samples = max(0, int(sample_rate * (payload.crossfade_ms or 0) / 1000))
    pending_tail: np.ndarray | None = None
    sent_pcm: list[bytes] = []
    chunks_sent = 0
    samples_sent = 0
    first_pcm_ready_seconds = 0.0
    first_chunk_sent_seconds = 0.0

    async def send_pcm(pcm: np.ndarray) -> None:
        nonlocal chunks_sent, samples_sent, first_chunk_sent_seconds
        data = pcm.astype(np.int16, copy=False).tobytes()
        if not data:
            return
        await websocket.send_bytes(data)
        sent_pcm.append(data)
        chunks_sent += 1
        samples_sent += len(pcm)
        if chunks_sent == 1:
            first_chunk_sent_seconds = time.time() - started_at

    def render_segment(segment_text: str) -> np.ndarray | None:
        segment_parts: list[np.ndarray] = []
        for result in model.generate(segment_text, **generate_kwargs):
            result_audio = getattr(result, "audio", None)
            if result_audio is None:
                continue
            pcm = _audio_to_pcm16le(result_audio)
            if pcm:
                segment_parts.append(np.frombuffer(pcm, dtype=np.int16))
        if not segment_parts:
            return None
        return (
            np.concatenate(segment_parts)
            if len(segment_parts) > 1
            else segment_parts[0]
        )

    for index, segment_text in enumerate(segments):
        # Run blocking inference off the event loop so already-rendered
        # segments reach the client while the next one is generating.
        segment_pcm = await asyncio.to_thread(render_segment, segment_text)
        if segment_pcm is None:
            continue
        if first_pcm_ready_seconds == 0.0:
            first_pcm_ready_seconds = time.time() - started_at

        segment_pcm = _trim_refelicia_pcm16_edges(
            segment_pcm,
            sample_rate,
            max_trim_ms=payload.trim_silence_ms or 0,
            threshold=payload.silence_threshold or 0,
            trim_start=index > 0,
            trim_end=index < len(segments) - 1,
        )
        if pending_tail is not None:
            segment_pcm = _crossfade_refelicia_pcm16(
                pending_tail, segment_pcm, overlap_samples
            )
            pending_tail = None

        is_last = index == len(segments) - 1
        if not is_last and overlap_samples > 0 and len(segment_pcm) > overlap_samples * 2:
            body = segment_pcm[:-overlap_samples]
            pending_tail = segment_pcm[-overlap_samples:]
        else:
            body = segment_pcm
        await send_pcm(body)

    if pending_tail is not None:
        await send_pcm(pending_tail)

    dump_path = os.getenv("MLX_AUDIO_REFELICIA_TTS_DUMP_WAV")
    if dump_path and sent_pcm:
        dump_audio = (
            np.frombuffer(b"".join(sent_pcm), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        audio_write(dump_path, dump_audio, sample_rate)

    generate_seconds = time.time() - started_at
    audio_seconds = samples_sent / sample_rate if sample_rate > 0 else 0.0
    await websocket.send_json(
        {
            "type": "end",
            "chunks": chunks_sent,
            "delivery_mode": payload.delivery_mode,
            "audio_seconds": audio_seconds,
            "generate_seconds": generate_seconds,
            "real_time_factor": generate_seconds / audio_seconds
            if audio_seconds > 0
            else 0.0,
            "first_pcm_ready_seconds": first_pcm_ready_seconds,
            "start_sent_seconds": start_sent_seconds,
            "first_chunk_sent_seconds": first_chunk_sent_seconds,
            "segments": len(segments),
        }
    )
    _log_refelicia_tts_request(
        mode=payload.delivery_mode or "streaming",
        text=payload.text,
        segments=len(segments),
        first_pcm_ready_seconds=first_pcm_ready_seconds,
        generate_seconds=generate_seconds,
        audio_seconds=audio_seconds,
    )


@app.websocket("/ws/tts")
async def refelicia_tts_websocket(websocket: WebSocket):
    """ReFelicia/SBV2-compatible TTS WebSocket.

    ReFeliciaV2 sends JSON requests and expects, per request:
    ``{"type":"start", ...}``, raw mono PCM16LE binary chunks, then
    ``{"type":"end", ...}``.
    """
    await websocket.accept()

    while True:
        try:
            payload = ReFeliciaTTSRequest.model_validate(
                await websocket.receive_json()
            )
            await _handle_refelicia_tts_request(websocket, payload)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                return


async def _stream_transcription(
    websocket: WebSocket,
    stt_model,
    audio_array: np.ndarray,
    sample_rate: int,
    language: Optional[str],
    is_partial: bool,
    streaming: bool = True,
):
    """Handle both streaming and non-streaming model inference over WebSocket.

    Streaming models (whose generate() accepts a ``stream`` parameter) receive
    the audio as an ``mx.array`` and yield token deltas sent as
    ``{"type": "delta", "delta": "..."}`` messages, followed by a
    ``{"type": "complete", ...}`` message.

    Non-streaming models fall back to temp-file + batch generate, sending the
    legacy ``{"text": ..., "is_partial": ...}`` format.
    """
    supports_stream = "stream" in inspect.signature(stt_model.generate).parameters

    if supports_stream and streaming:
        result_iter = stt_model.generate(
            mx.array(audio_array), stream=True, language=language, verbose=False
        )
        accumulated = ""
        detected_language = language
        for chunk in result_iter:
            delta = (
                chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
            )
            accumulated += delta
            # Pick up detected language from streaming results
            chunk_lang = getattr(chunk, "language", None)
            if chunk_lang and detected_language is None:
                detected_language = chunk_lang
            await websocket.send_json({"type": "delta", "delta": delta})

        await websocket.send_json(
            {
                "type": "complete",
                "text": accumulated,
                "segments": None,
                "language": detected_language,
                "is_partial": is_partial,
            }
        )
    else:
        tmp_path = f"/tmp/realtime_{time.time()}.mp3"
        audio_write(tmp_path, audio_array, sample_rate)
        try:
            result = stt_model.generate(tmp_path, language=language, verbose=False)
            segments = (
                sanitize_for_json(result.segments)
                if hasattr(result, "segments") and result.segments
                else None
            )
            await websocket.send_json(
                {
                    "text": result.text,
                    "segments": segments,
                    "language": getattr(result, "language", language),
                    "is_partial": is_partial,
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@app.websocket("/v1/audio/transcriptions/realtime")
async def stt_realtime_transcriptions(websocket: WebSocket):
    """Realtime transcription via WebSocket."""
    await websocket.accept()

    try:
        # Receive initial configuration
        config = await websocket.receive_json()
        model_name = config.get(
            "model", "mlx-community/whisper-large-v3-turbo-asr-fp16"
        )
        language = config.get("language", None)
        sample_rate = config.get("sample_rate", 16000)
        streaming = config.get("streaming", True)

        print(
            f"Configuration received: model={model_name}, language={language}, sample_rate={sample_rate}, streaming={streaming}"
        )

        # Load the STT model
        print("Loading STT model...")
        stt_model = model_provider.load_model(model_name)
        print("STT model loaded successfully")

        # Initialize WebRTC VAD for speech detection
        vad = webrtcvad.Vad(
            3
        )  # Mode 3 is most aggressive (0-3, higher = more aggressive)
        # VAD requires specific frame sizes: 10ms, 20ms, or 30ms at 8kHz, 16kHz, 32kHz, or 48kHz
        vad_frame_duration_ms = 30  # 30ms frames
        vad_frame_size = int(sample_rate * vad_frame_duration_ms / 1000)
        print(
            f"VAD initialized: frame_size={vad_frame_size} samples ({vad_frame_duration_ms}ms at {sample_rate}Hz)"
        )

        # Buffer for accumulating audio chunks with speech
        audio_buffer = []
        min_chunk_size = int(sample_rate * 0.5)  # Minimum 0.5 seconds before processing
        initial_chunk_size = int(
            sample_rate * 1.5
        )  # Process first 1.5 seconds for real-time feedback
        max_chunk_size = int(
            sample_rate * 5.0
        )  # Maximum 10 seconds to avoid memory issues
        silence_skip_count = 0
        speech_chunk_count = 0
        last_speech_time = time.time()  # Track when we last detected speech
        silence_threshold_seconds = 0.5  # Process when silence > 0.5 seconds
        initial_chunk_processed = False  # Track if we've processed the initial chunk

        await websocket.send_json({"status": "ready", "message": "Ready to transcribe"})
        print("Ready to transcribe")

        while True:
            # Receive message
            try:
                message = await websocket.receive()
            except Exception:
                break

            if "bytes" in message:
                # Audio data received as int16
                audio_chunk_int16 = np.frombuffer(message["bytes"], dtype=np.int16)

                # Process audio in VAD frame sizes to detect speech
                # WebRTC VAD requires frames of exactly 10ms, 20ms, or 30ms
                # at sample rates of 8000, 16000, 32000, or 48000 Hz
                num_frames = len(audio_chunk_int16) // vad_frame_size
                has_speech = False
                speech_frames = 0

                # Check each VAD frame for speech activity
                for i in range(num_frames):
                    frame_start = i * vad_frame_size
                    frame_end = frame_start + vad_frame_size
                    frame = audio_chunk_int16[frame_start:frame_end]

                    # VAD requires exact frame size
                    if len(frame) == vad_frame_size:
                        try:
                            if vad.is_speech(frame.tobytes(), sample_rate):
                                has_speech = True
                                speech_frames += 1
                        except (ValueError, OSError) as e:
                            # If VAD fails (wrong sample rate or frame size), assume speech (conservative)
                            # This can happen if sample rate doesn't match VAD requirements
                            print(f"VAD error (assuming speech): {e}")
                            has_speech = True
                            speech_frames += 1

                # Handle remaining samples that don't form a complete frame
                # These will be processed in the next chunk

                # Only accumulate audio if it contains speech
                current_time = time.time()
                if has_speech:
                    # Convert to float32 for buffer
                    audio_chunk_float = audio_chunk_int16.astype(np.float32) / 32768.0
                    audio_buffer.extend(audio_chunk_float)
                    speech_chunk_count += 1
                    silence_skip_count = 0
                    last_speech_time = current_time

                    if len(audio_buffer) % (sample_rate * 2) < len(audio_chunk_float):
                        # Log every ~2 seconds of buffer
                        print(
                            f"Speech detected ({speech_frames}/{num_frames} frames): buffer {len(audio_buffer)} samples ({len(audio_buffer)/sample_rate:.2f}s)"
                        )
                else:
                    silence_skip_count += 1
                    # Only log silence periodically to reduce noise
                    if silence_skip_count % 20 == 0:
                        print(f"Silence detected: skipped {silence_skip_count} chunks")

                # Determine if we should process:
                # 1. Process initial chunk (first 1.5s) for real-time feedback while accumulating
                # 2. If we have silence > 0.5 seconds and buffer has speech (end of utterance)
                # 3. If buffer reaches maximum size (to avoid memory issues)
                time_since_last_speech = current_time - last_speech_time
                should_process_initial = False
                should_process_final = False

                if len(audio_buffer) > 0:
                    # Process initial chunk for real-time feedback (only once per speech segment)
                    if (
                        not initial_chunk_processed
                        and len(audio_buffer) >= initial_chunk_size
                        and has_speech  # Only if we're still detecting speech
                    ):
                        should_process_initial = True
                        print(
                            f"Processing initial chunk for real-time feedback: {initial_chunk_size/sample_rate:.2f}s, total buffer: {len(audio_buffer)/sample_rate:.2f}s"
                        )
                    # Process if we have enough silence after speech (end of utterance)
                    elif (
                        time_since_last_speech >= silence_threshold_seconds
                        and len(audio_buffer) >= min_chunk_size
                    ):
                        should_process_final = True
                        print(
                            f"Processing due to silence gap: {time_since_last_speech:.2f}s silence, buffer: {len(audio_buffer)/sample_rate:.2f}s"
                        )
                    # Or if buffer is getting too large (continuous speech)
                    elif len(audio_buffer) >= max_chunk_size:
                        should_process_final = True
                        print(
                            f"Processing due to max buffer size: {len(audio_buffer)/sample_rate:.2f}s"
                        )

                # Process initial chunk for real-time feedback
                if should_process_initial and len(audio_buffer) >= initial_chunk_size:
                    process_size = initial_chunk_size
                    audio_array = np.array(audio_buffer[:process_size])
                    initial_chunk_processed = True

                    try:
                        await _stream_transcription(
                            websocket,
                            stt_model,
                            audio_array,
                            sample_rate,
                            language,
                            is_partial=True,
                            streaming=streaming,
                        )
                    except Exception as e:
                        import traceback

                        error_msg = str(e)
                        traceback.print_exc()
                        print(f"Error during initial transcription: {error_msg}")
                        await websocket.send_json(
                            {"error": error_msg, "status": "error"}
                        )

                # Process final chunk (entire accumulated buffer)
                if should_process_final and len(audio_buffer) > 0:
                    # Process the entire buffer (continuous speech chunk)
                    process_size = len(audio_buffer)
                    audio_array = np.array(audio_buffer)

                    try:
                        await _stream_transcription(
                            websocket,
                            stt_model,
                            audio_array,
                            sample_rate,
                            language,
                            is_partial=False,
                            streaming=streaming,
                        )

                        # Clear processed audio from buffer and reset state
                        audio_buffer = []
                        initial_chunk_processed = False
                        print(
                            f"Processed final chunk: {process_size} samples ({process_size/sample_rate:.2f}s), buffer cleared"
                        )

                    except Exception as e:
                        import traceback

                        error_msg = str(e)
                        traceback.print_exc()
                        print(f"Error during transcription: {error_msg}")
                        await websocket.send_json(
                            {"error": error_msg, "status": "error"}
                        )

            elif "text" in message:
                # JSON message received (e.g., stop command)
                try:
                    data = json.loads(message["text"])
                    if data.get("action") == "stop":
                        break
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e), "status": "error"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# OpenAI Realtime clients send PCM at 24 kHz by default, so we assume that
# until the client declares a different ``audio.input.format.rate``. The
# model-side rate is read off the streaming session's ``input_sample_rate``
# attribute; the server resamples client → model on ingress.
_REALTIME_DEFAULT_CLIENT_RATE = 24000


def _default_transcription_delay_ms() -> Optional[int]:
    raw = os.getenv("MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _open_streaming_session(model, *, temperature: float, delay_ms: Optional[int]):
    """Open a streaming session, forwarding ``transcription_delay_ms`` only to
    models that declare the parameter.
    """
    kwargs: dict = {"temperature": temperature}
    if delay_ms is not None:
        sig = inspect.signature(model.create_streaming_session)
        if "transcription_delay_ms" in sig.parameters:
            kwargs["transcription_delay_ms"] = delay_ms
    return model.create_streaming_session(**kwargs)


def _resample_pcm16_to_rate(
    pcm16: np.ndarray, from_rate: int, to_rate: int
) -> np.ndarray:
    """Linear resample int16 PCM to ``to_rate``.

    Returns float32 samples in [-1, 1]. If the rates already match, this is
    a plain dtype cast.
    """
    samples = pcm16.astype(np.float32) / 32768.0
    if from_rate == to_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * to_rate / from_rate))
    if n_out <= 1:
        return samples[:n_out].astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _resolve_realtime_model_name(requested_model: Optional[str]) -> Optional[str]:
    """Return the requested model ID, or the server-configured default.

    The default is read from ``$MLX_AUDIO_REALTIME_MODEL`` (settable via the
    ``--realtime-model`` CLI flag). Returns ``None`` when neither the client
    nor the server supplies a model — the caller is expected to reject the
    connection with a clear error.
    """
    normalized = (requested_model or "").strip()
    if normalized:
        return normalized
    configured = (os.getenv("MLX_AUDIO_REALTIME_MODEL") or "").strip()
    return configured or None


_REALTIME_VAD_MODEL_DEFAULT: str = "mlx-community/silero-vad"
_realtime_vad_models: Dict[str, Any] = {}


def _resolve_vad_model_name() -> str:
    """Return the VAD model used for server-side turn detection.

    Read at request time from ``$MLX_AUDIO_VAD_MODEL`` (settable via the
    ``--vad-model`` CLI flag), defaulting to Silero.
    """
    return (
        os.getenv("MLX_AUDIO_VAD_MODEL") or ""
    ).strip() or _REALTIME_VAD_MODEL_DEFAULT


def _load_realtime_vad_model(name: str):
    """Load and cache a streaming VAD model for server-side turn detection.

    Kept behind a module-level indirection so the realtime endpoint can be
    exercised with a fake VAD model in tests without hitting Hugging Face.
    """
    model = _realtime_vad_models.get(name)
    if model is None:
        from mlx_audio.vad import load as load_vad

        model = load_vad(name)
        _realtime_vad_models[name] = model
    return model


@app.websocket("/v1/realtime")
async def realtime_ws(websocket: WebSocket):
    """OpenAI Realtime API-compatible WebSocket endpoint.

    Works with any STT model whose ``load_model`` result exposes the
    streaming-session protocol. A streaming session must provide:
      - ``feed(samples: np.ndarray[float32])``
      - ``close()``
      - ``step(max_decode_tokens: int) -> list[str]``
      - ``done: bool``
      - ``input_sample_rate: int``  (the native rate expected by ``feed``)
    Models that support an adjustable transcription-delay / latency knob
    should accept ``transcription_delay_ms: Optional[int]`` on
    ``create_streaming_session``; models without the concept can ignore it.

    Client → server events follow OpenAI Realtime (subset):
      ``session.update``, ``input_audio_buffer.append``,
      ``input_audio_buffer.commit``.
    Server → client events: ``session.created`` / ``session.updated``,
    ``conversation.item.added``, ``input_audio_buffer.committed``,
    ``conversation.item.input_audio_transcription.delta`` /
    ``.completed``, and ``error``.

    Turn detection follows OpenAI's ``turn_detection`` on
    ``session.audio.input``. With ``{"type": "server_vad"}`` the server runs a
    streaming VAD (Silero by default, see ``--vad-model``), emits
    ``input_audio_buffer.speech_started`` / ``input_audio_buffer.speech_stopped``
    and auto-commits each turn — the client never sends
    ``input_audio_buffer.commit``. With ``null`` (the default) the client drives
    commits manually. ``semantic_vad`` is not implemented yet.

    Model is selected via ``?model=<id>`` or ``session.update.model``;
    defaults to ``$MLX_AUDIO_REALTIME_MODEL``.
    """
    await websocket.accept()

    def _new_event_id() -> str:
        return f"event_{uuid.uuid4().hex[:16]}"

    async def send_event(payload: dict):
        payload = {"event_id": _new_event_id(), **payload}
        await websocket.send_json(payload)

    async def send_error(message: str):
        await send_event({"type": "error", "error": {"message": message}})

    requested_model = websocket.query_params.get("model")
    model_name = _resolve_realtime_model_name(requested_model)
    if model_name is None:
        await send_error(
            "no realtime model configured: pass ?model=<id>, set "
            "session.update.model, or start the server with --realtime-model"
        )
        await websocket.close()
        return
    try:
        model = model_provider.load_model(model_name)
    except Exception as e:
        await send_error(f"load failed: {e}")
        await websocket.close()
        return

    if not hasattr(model, "create_streaming_session"):
        await send_error(f"model {model_name!r} does not support streaming")
        await websocket.close()
        return

    temperature = 0.0
    transcription_delay_ms = _default_transcription_delay_ms()
    session = _open_streaming_session(
        model, temperature=temperature, delay_ms=transcription_delay_ms
    )
    full_text_parts: list[str] = []
    current_item_id: Optional[str] = None
    client_input_rate = _REALTIME_DEFAULT_CLIENT_RATE
    turn_config: Optional[ServerVadConfig] = None
    turn_detector: Optional[StreamingVad] = None

    def _new_item_id() -> str:
        return f"item_{uuid.uuid4().hex[:16]}"

    async def drain_deltas(max_decode_tokens: int = 8) -> bool:
        """Run one session.step, ship deltas. Returns session.done.

        MLX runs inline on the event-loop thread (not via a worker thread):
        MLX streams are thread-bound, so all realtime MLX work — the
        transcription step and the VAD — must share one thread, otherwise you
        hit "no Stream(gpu, N) in current thread".
        """
        async with REALTIME_INFERENCE_LOCK:
            deltas = session.step(max_decode_tokens=max_decode_tokens)
        for delta in deltas:
            full_text_parts.append(delta)
            await send_event(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": current_item_id,
                    "content_index": 0,
                    "delta": delta,
                }
            )
        return session.done

    async def send_done():
        text = "".join(full_text_parts)
        await send_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": current_item_id,
                "content_index": 0,
                "transcript": text,
            }
        )

    async def finalize_turn() -> None:
        """Commit the current turn: emit ``committed``, drain the remaining
        transcription, emit ``completed``, then reopen a fresh session.

        Shared by the manual ``input_audio_buffer.commit`` path and the
        server-VAD auto-commit path.
        """
        nonlocal session, full_text_parts, current_item_id
        await send_event(
            {
                "type": "input_audio_buffer.committed",
                "item_id": current_item_id,
                "previous_item_id": None,
            }
        )
        async with REALTIME_INFERENCE_LOCK:
            session.close()
        while not await drain_deltas(max_decode_tokens=16):
            pass
        await send_done()
        async with REALTIME_INFERENCE_LOCK:
            session = _open_streaming_session(
                model, temperature=temperature, delay_ms=transcription_delay_ms
            )
        full_text_parts = []
        current_item_id = None

    session_id = f"sess_{uuid.uuid4().hex[:16]}"

    def _session_snapshot() -> dict:
        return {
            "id": session_id,
            "object": "realtime.session",
            "type": "transcription",
            "model": model_name,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": client_input_rate},
                    "transcription": {"model": model_name},
                    "turn_detection": turn_config.to_dict() if turn_config else None,
                }
            },
        }

    await send_event({"type": "session.created", "session": _session_snapshot()})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_error("invalid JSON")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "session.update":
                session_payload = msg.get("session") or {}
                audio_input = (session_payload.get("audio") or {}).get("input") or {}
                transcription_cfg = audio_input.get("transcription") or {}
                resolved_model = session_payload.get("model") or transcription_cfg.get(
                    "model"
                )
                fmt = audio_input.get("format") or {}
                requested_rate = fmt.get("rate")
                if isinstance(requested_rate, int) and requested_rate > 0:
                    client_input_rate = requested_rate
                target_model_name = _resolve_realtime_model_name(resolved_model)
                if target_model_name and target_model_name != model_name:
                    try:
                        model = model_provider.load_model(target_model_name)
                    except Exception as e:
                        await send_error(f"load failed: {e}")
                        continue
                    if not hasattr(model, "create_streaming_session"):
                        await send_error(
                            f"model {target_model_name!r} does not support streaming"
                        )
                        continue
                    model_name = target_model_name
                    async with REALTIME_INFERENCE_LOCK:
                        session = _open_streaming_session(
                            model,
                            temperature=temperature,
                            delay_ms=transcription_delay_ms,
                        )
                    full_text_parts = []
                    current_item_id = None

                if "turn_detection" in audio_input:
                    try:
                        turn_config = parse_turn_detection(
                            audio_input.get("turn_detection")
                        )
                    except TurnDetectionError as e:
                        await send_error(str(e))
                        continue
                    if turn_config is None:
                        turn_detector = None
                    else:
                        try:
                            vad_model = _load_realtime_vad_model(
                                _resolve_vad_model_name()
                            )
                        except Exception as e:
                            turn_config = None
                            turn_detector = None
                            await send_error(f"vad load failed: {e}")
                            continue
                        turn_detector = StreamingVad(vad_model, turn_config)

                await send_event(
                    {"type": "session.updated", "session": _session_snapshot()}
                )

            elif msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    continue
                # In server-VAD mode the conversation item is created on
                # ``speech_started`` (below) to match OpenAI; in manual-commit
                # mode it is created on the first appended audio.
                if current_item_id is None and turn_detector is None:
                    current_item_id = _new_item_id()
                    await send_event(
                        {
                            "type": "conversation.item.added",
                            "item": {
                                "id": current_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_audio"}],
                            },
                        }
                    )
                pcm16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                samples = _resample_pcm16_to_rate(
                    pcm16, client_input_rate, session.input_sample_rate
                )
                async with REALTIME_INFERENCE_LOCK:
                    session.feed(samples)
                # Opportunistic draining between chunks so deltas flow early.
                await drain_deltas(max_decode_tokens=8)

                if turn_detector is not None:
                    vad_samples = _resample_pcm16_to_rate(
                        pcm16, client_input_rate, VAD_SAMPLE_RATE
                    )
                    async with REALTIME_INFERENCE_LOCK:
                        # Inline (see drain_deltas): VAD MLX must share the
                        # transcription thread, or MLX streams collide.
                        turn_events = turn_detector.process(vad_samples)
                    for turn_event in turn_events:
                        if turn_event.kind is TurnEventKind.SPEECH_STARTED:
                            if current_item_id is None:
                                current_item_id = _new_item_id()
                                await send_event(
                                    {
                                        "type": "conversation.item.added",
                                        "item": {
                                            "id": current_item_id,
                                            "object": "realtime.item",
                                            "type": "message",
                                            "role": "user",
                                            "content": [{"type": "input_audio"}],
                                        },
                                    }
                                )
                            await send_event(
                                {
                                    "type": "input_audio_buffer.speech_started",
                                    "audio_start_ms": turn_event.audio_ms,
                                    "item_id": current_item_id,
                                }
                            )
                        else:
                            await send_event(
                                {
                                    "type": "input_audio_buffer.speech_stopped",
                                    "audio_end_ms": turn_event.audio_ms,
                                    "item_id": current_item_id,
                                }
                            )
                            await finalize_turn()
                            turn_detector.reset_turn()

            elif msg_type == "input_audio_buffer.commit":
                if current_item_id is None:
                    current_item_id = _new_item_id()
                    await send_event(
                        {
                            "type": "conversation.item.added",
                            "item": {
                                "id": current_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_audio"}],
                            },
                        }
                    )
                await finalize_turn()
                if turn_detector is not None:
                    turn_detector.reset_turn()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback

        traceback.print_exc()
        try:
            await send_error(str(e))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        mx.clear_cache()


class MLXAudioStudioServer:
    def __init__(self, start_ui=False, log_dir="logs"):
        self.start_ui = start_ui
        self.ui_process = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def start_ui_background(self):
        """Start UI with logs redirected to file"""
        ui_path = Path(__file__).parent / "ui"

        try:
            # Install deps silently
            subprocess.run(
                ["npm", "install"],
                cwd=str(ui_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            raise Exception(
                "✗ Error: 'npm' is not installed or not found in PATH. UI will not start."
            )
        except subprocess.CalledProcessError as e:
            raise Exception("✗ Error running 'npm install':\n", e)

        try:
            # Start UI with logs to file
            ui_log = open(self.log_dir / "ui.log", "w")
            self.ui_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_path),
                stdout=ui_log,
                stderr=subprocess.STDOUT,
            )
            print(f"✓ UI started (logs: {self.log_dir}/ui.log)")
        except FileNotFoundError:
            raise Exception(
                "✗ Error: 'npm' is not installed or not found in PATH. UI server not started."
            )
        except Exception as e:
            raise Exception(f"✗ Failed to start UI: {e}")

    def start_server(self, host="localhost", port=8000, reload=False, realtime=False):
        if self.start_ui:
            self.start_ui_background()
            time.sleep(2)
            webbrowser.open("http://localhost:3000")
            print(f"✓ API server starting on http://{host}:{port}")
            print("✓ Studio UI available at http://localhost:3000")
            print("\nPress Ctrl+C to stop both servers")
        elif realtime:
            print(f"✓ Realtime server starting on http://{host}:{port}")
            print("✓ Standard endpoints remain mounted; prefer realtime endpoints.")

        try:
            uvicorn.run(
                "mlx_audio.server:app",
                host=host,
                port=port,
                reload=reload,
                workers=1,
                loop="asyncio",
            )
        finally:
            if self.ui_process:
                self.ui_process.terminate()
                print("✓ UI server stopped")

            ui_log_path = self.log_dir / "ui.log"
            if ui_log_path.exists():
                ui_log_path.unlink()
                print(f"✓ UI logs deleted from {ui_log_path}")


def main():
    parser = argparse.ArgumentParser(description="MLX Audio API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server.",
    )
    parser.add_argument(
        "--start-ui",
        action="store_true",
        help="Start the Studio UI alongside the API server",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save server logs",
    )
    parser.add_argument(
        "--realtime-model",
        type=str,
        default=None,
        help=(
            "Default model for /v1/realtime when the client omits ?model=. "
            "Overrides $MLX_AUDIO_REALTIME_MODEL."
        ),
    )
    parser.add_argument(
        "--realtime-transcription-delay-ms",
        type=int,
        default=None,
        help=(
            "Transcription latency/quality knob for streaming STT models that "
            "expose a ``transcription_delay_ms`` parameter (e.g. voxtral_realtime). "
            "Lower values reduce latency at the cost of accuracy. When unset, "
            "each model uses its own default. Overrides "
            "$MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS."
        ),
    )
    parser.add_argument(
        "--vad-model",
        type=str,
        default=None,
        help=(
            "Streaming VAD model used for server-side turn detection on "
            "/v1/realtime (server_vad). Overrides $MLX_AUDIO_VAD_MODEL "
            "(default: mlx-community/silero-vad)."
        ),
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Start the server for /v1/realtime usage.",
    )
    parser.add_argument(
        "--tts-max-batch-size",
        type=int,
        default=None,
        help=(
            "Maximum compatible TTS speech requests per continuous batch session. "
            "Overrides $MLX_AUDIO_TTS_MAX_BATCH_SIZE."
        ),
    )

    args = parser.parse_args()
    if args.realtime_model:
        os.environ["MLX_AUDIO_REALTIME_MODEL"] = args.realtime_model
    if args.realtime_transcription_delay_ms is not None:
        os.environ["MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS"] = str(
            args.realtime_transcription_delay_ms
        )
    if args.vad_model:
        os.environ["MLX_AUDIO_VAD_MODEL"] = args.vad_model
    if args.tts_max_batch_size is not None:
        os.environ["MLX_AUDIO_TTS_MAX_BATCH_SIZE"] = str(args.tts_max_batch_size)

    setup_cors(app, args.allowed_origins)

    client = MLXAudioStudioServer(start_ui=args.start_ui, log_dir=args.log_dir)
    client.start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
