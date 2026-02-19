import asyncio
import json
import os
import sys

from pydantic import BaseModel
from tqdm import tqdm
from typing import TypeVar

from semantic_navigator.models import (
    Aspect, AspectPool, Facets, Label, Labels,
)
from semantic_navigator.util import extract_json, repair_json
from semantic_navigator.gpu import (
    detect_device_memory, select_best_gguf, _resolve_gpu_layers,
)

T = TypeVar("T", bound=BaseModel)

max_retries = 60
max_count_retries = 5


def _build_labels_schema(count: int | None) -> dict:
    label_schema = {
        "type": "object",
        "properties": {
            "overarchingTheme": {"type": "string"},
            "distinguishingFeature": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["overarchingTheme", "distinguishingFeature", "label"],
    }
    array_schema: dict = {"type": "array", "items": label_schema}
    if count is not None:
        array_schema["minItems"] = count
        array_schema["maxItems"] = count
    return {
        "type": "object",
        "properties": {"labels": array_schema},
        "required": ["labels"],
    }


def _local_complete(model: object, prompt: str, response_format: dict | None = None) -> str:
    kwargs: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    result = model.create_chat_completion(**kwargs)
    return result["choices"][0]["message"]["content"]


async def _invoke_openai(aspect: Aspect, prompt: str, output_type: type[T]) -> T:
    response = await aspect.openai_client.responses.parse(
        model = aspect.openai_model,
        input = prompt,
        text_format = output_type,
    )
    if response.output_parsed is None:
        raise ValueError("OpenAI returned no parsed output")
    return response.output_parsed


async def _invoke_local(aspect: Aspect, prompt: str, expected_count: int | None) -> str:
    response_format = None
    if expected_count is not None:
        response_format = {
            "type": "json_object",
            "schema": _build_labels_schema(expected_count),
        }
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _local_complete, aspect.local_model, prompt, response_format)


def _parse_raw_response(raw: str, output_type: type[T], debug: bool) -> T:
    extracted = extract_json(raw)
    if debug:
        print(f"[debug] extracted JSON: {extracted[:500]}{'...' if len(extracted) > 500 else ''}")
    try:
        return output_type.model_validate_json(extracted)
    except Exception:
        repaired = repair_json(extracted)
        if debug:
            print(f"[debug] repaired JSON: {repaired[:500]}{'...' if len(repaired) > 500 else ''}")
        return output_type.model_validate_json(repaired)


def _validate_label_count(parsed: T, expected_count: int | None, aspect_name: str, attempt: int) -> T | None:
    if expected_count is None or not hasattr(parsed, 'labels') or len(parsed.labels) == expected_count:
        return parsed
    if len(parsed.labels) > expected_count:
        tqdm.write(f"[{aspect_name}] Truncating {len(parsed.labels)} → {expected_count} labels")
        parsed.labels = parsed.labels[:expected_count]
        return parsed
    if attempt < max_count_retries - 1:
        tqdm.write(f"[{aspect_name}] Retry {attempt + 1}/{max_count_retries}: expected {expected_count} labels, got {len(parsed.labels)}")
        return None
    tqdm.write(f"Padding {len(parsed.labels)} labels to {expected_count} (gave up after {max_count_retries} attempts)")
    while len(parsed.labels) < expected_count:
        parsed.labels.append(Label(
            overarchingTheme = "Miscellaneous",
            distinguishingFeature = "Ungrouped",
            label = "Miscellaneous",
        ))
    return parsed


async def complete(facets: Facets, prompt: str, output_type: type[T], progress: tqdm | None = None, expected_count: int | None = None) -> T:
    for attempt in range(max_retries):
        if facets.debug:
            print(f"\n[debug] prompt ({len(prompt)} chars):\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        aspect = await facets.pool.acquire()
        release_aspect = True
        try:
            if aspect.openai_client is not None:
                parsed = await _invoke_openai(aspect, prompt, output_type)
            elif aspect.local_model is not None:
                try:
                    raw = await _invoke_local(aspect, prompt, expected_count)
                except (OSError, RuntimeError) as e:
                    release_aspect = False
                    remaining = facets.pool.remove(aspect)
                    if not remaining:
                        raise SystemExit(f"All inference backends have failed. Last error: {e}")
                    tqdm.write(f"Aspect {aspect.name} failed permanently ({e}), removed from pool ({remaining} remaining)")
                    if attempt < max_retries - 1:
                        continue
                    raise
                if facets.debug:
                    print(f"[debug] raw output ({len(raw)} chars):\n{raw[:500]}{'...' if len(raw) > 500 else ''}")
                parsed = _parse_raw_response(raw, output_type, facets.debug)
            else:
                raise ValueError(f"Aspect {aspect.name} has no backend")

            validated = _validate_label_count(parsed, expected_count, aspect.name, attempt)
            if validated is None:
                continue
            if progress is not None:
                progress.update(1)
            return validated
        except (ValueError, RuntimeError) as e:
            if attempt < max_retries - 1:
                tqdm.write(f"[{aspect.name}] Retry {attempt + 1}/{max_retries}: {e}")
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"[{aspect.name}] Retry {attempt + 1}/{max_retries}: parse error: {e}")
                continue
            raise
        finally:
            if release_aspect:
                facets.pool.release(aspect)


def _create_local_model(local: str, local_file: str | None, gpu: bool, device: int, gpu_layers: int | None, n_ctx: int, debug: bool) -> object:
    from llama_cpp import Llama
    from semantic_navigator.util import case_insensitive_glob

    is_local_file = os.path.exists(local) or "\\" in local or local.count("/") > 1

    if is_local_file:
        model_size = os.path.getsize(local)
        n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device, model_size)
        print(f"Local model: {local} ({'GPU ' + str(device) if gpu else 'CPU'})")
        return Llama(
            model_path = local,
            n_gpu_layers = n_gpu_layers,
            n_ctx = n_ctx,
            verbose = debug,
        )

    if local_file is None:
        memory_budget = detect_device_memory(gpu, device)
        kv_cache_reserve = n_ctx * 128 * 2 + int(1e9)
        print(f"Querying HuggingFace for available quantizations...", flush=True)
        local_files, model_size = select_best_gguf(local, memory_budget, kv_cache_reserve)
        n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device, model_size)
        print(f"Local model: {local} (auto-selected: {local_files[0]}, {model_size / 1e9:.1f} GB, {'GPU ' + str(device) if gpu else 'CPU'})")
    else:
        local_files = [case_insensitive_glob(local_file)]
        n_gpu_layers = _resolve_gpu_layers(gpu, gpu_layers, device)
        print(f"Local model: {local} (file: {local_files[0]}, {'GPU ' + str(device) if gpu else 'CPU'})")
    return Llama.from_pretrained(
        repo_id = local,
        filename = local_files[0],
        additional_files = local_files[1:] or None,
        n_gpu_layers = n_gpu_layers,
        n_ctx = n_ctx,
        verbose = debug,
    )


def _create_local_aspects(local: str, local_file: str | None, gpu: bool, devices: list[int], gpu_layers: int | None, n_ctx: int, debug: bool) -> list[Aspect]:
    aspects: list[Aspect] = []
    if gpu:
        for device in devices:
            vram = detect_device_memory(True, device)
            if vram is not None:
                # Quick size check
                pass
            try:
                model = _create_local_model(local, local_file, True, device, gpu_layers, n_ctx, debug)
            except (ValueError, RuntimeError) as e:
                print(f"Skipping GPU {device}: failed to load model ({e})")
                continue
            aspects.append(Aspect(
                name = f"local/gpu:{device}",
                cli_command = None,
                local_model = model,
                local_n_ctx = model.n_ctx(),
                openai_client = None,
                openai_model = None,
            ))
    else:
        model = _create_local_model(local, local_file, False, 0, 0, n_ctx, debug)
        aspects.append(Aspect(
            name = "local/cpu",
            cli_command = None,
            local_model = model,
            local_n_ctx = model.n_ctx(),
            openai_client = None,
            openai_model = None,
        ))
    return aspects


def _create_embedding_model(openai_embedding_model: str | None, embedding_model: str, gpu: bool, devices: list[int], cpu_offload: bool, batch_size: int | None) -> tuple:
    from fastembed import TextEmbedding

    if openai_embedding_model is not None:
        from openai import AsyncOpenAI
        openai_client_for_embed = AsyncOpenAI()
        print(f"OpenAI embedding model: {openai_embedding_model}")
        if batch_size is None:
            batch_size = 256
        return None, openai_client_for_embed, openai_embedding_model, batch_size

    if gpu:
        device = devices[0]
        if sys.platform == "win32":
            providers = [("DmlExecutionProvider", {"device_id": device})]
        elif sys.platform == "darwin":
            providers = ["CoreMLExecutionProvider"]
        else:
            providers = [("CUDAExecutionProvider", {"device_id": str(device)})]
        if cpu_offload:
            providers.append("CPUExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
    if batch_size is None:
        batch_size = 32 if gpu else 256
    embedding = TextEmbedding(model_name = embedding_model, providers = providers)
    return embedding, None, None, batch_size


def initialize(repository: str, model_identity: str, local: str | None, local_file: str | None, embedding_model: str, gpu: bool, cpu_offload: bool, devices: list[int], gpu_layers: int | None, batch_size: int | None, n_ctx: int, timeout: int, debug: bool, openai_model: str | None = None, openai_embedding_model: str | None = None) -> Facets:
    aspects: list[Aspect] = []

    if local is not None:
        aspects.extend(_create_local_aspects(local, local_file, gpu, devices, gpu_layers, n_ctx, debug))
    if openai_model is not None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        print(f"OpenAI model: {openai_model}")
        aspects.append(Aspect(
            name = "openai/0",
            cli_command = None,
            local_model = None,
            local_n_ctx = None,
            openai_client = client,
            openai_model = openai_model,
        ))

    if not aspects:
        raise SystemExit("Error: no inference backends available.")

    embedding, openai_client_for_embed, _, resolved_batch_size = _create_embedding_model(
        openai_embedding_model, embedding_model, gpu, devices, cpu_offload, batch_size,
    )

    print(f"Initialized {len(aspects)} aspect{'s' if len(aspects) != 1 else ''}: {', '.join(a.name for a in aspects)}")
    return Facets(
        repository = repository,
        model_identity = model_identity,
        pool = AspectPool(aspects),
        embedding_model = embedding,
        embedding_model_name = embedding_model,
        gpu = gpu,
        batch_size = resolved_batch_size,
        timeout = timeout,
        debug = debug,
        openai_client = openai_client_for_embed,
        openai_embedding_model = openai_embedding_model,
    )
