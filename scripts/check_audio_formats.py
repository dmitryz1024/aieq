from __future__ import annotations

import argparse
from typing import Iterable

import sounddevice as sd

SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000, 384000)
DTYPES = ("float32", "int32", "int24", "int16")


def device_label(index: int, raw: dict, hostapis: list[dict]) -> str:
    hostapi = hostapis[int(raw["hostapi"])]["name"]
    return f"{index}: [{hostapi}] {raw['name']}"


def matching_device_indexes(kind: str, query: str | None) -> list[int]:
    devices = sd.query_devices()
    if query is None:
        return []
    try:
        return [int(query)]
    except ValueError:
        needle = query.casefold()
    result: list[int] = []
    for index, raw in enumerate(devices):
        channels = int(raw["max_input_channels"] if kind == "input" else raw["max_output_channels"])
        if channels > 0 and needle in str(raw["name"]).casefold():
            result.append(index)
    return result


def supported_rates_for_device(index: int, *, kind: str, dtype: str) -> list[int]:
    result: list[int] = []
    checker = sd.check_input_settings if kind == "input" else sd.check_output_settings
    for rate in SAMPLE_RATES:
        try:
            checker(device=index, channels=2, dtype=dtype, samplerate=float(rate))
        except Exception:
            try:
                checker(device=index, channels=1, dtype=dtype, samplerate=float(rate))
            except Exception:
                continue
        result.append(rate)
    return result


def supported_duplex_settings(input_index: int, output_index: int) -> list[tuple[int, str]]:
    input_device = sd.query_devices(input_index)
    output_device = sd.query_devices(output_index)
    channels = max(1, min(2, int(input_device["max_input_channels"]), int(output_device["max_output_channels"])))
    result: list[tuple[int, str]] = []
    for rate in SAMPLE_RATES:
        for dtype in DTYPES:
            try:
                sd.check_input_settings(
                    device=input_index,
                    channels=channels,
                    dtype=dtype,
                    samplerate=float(rate),
                )
                sd.check_output_settings(
                    device=output_index,
                    channels=channels,
                    dtype=dtype,
                    samplerate=float(rate),
                )
            except Exception:
                continue
            result.append((rate, dtype))
    return result


def print_device_list(kind: str, indexes: Iterable[int] | None = None) -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    selected = set(indexes or [])
    print(f"\n{kind.upper()} devices")
    for index, raw in enumerate(devices):
        channels = int(raw["max_input_channels"] if kind == "input" else raw["max_output_channels"])
        if channels <= 0:
            continue
        marker = "*" if index in selected else " "
        default_rate = int(float(raw.get("default_samplerate") or 0))
        print(f"{marker} {device_label(index, raw, hostapis)} | channels={channels}, default={default_rate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PortAudio/sounddevice sample-rate support for AIEQ.")
    parser.add_argument("--input", help="Input device index or part of its name")
    parser.add_argument("--output", help="Output device index or part of its name")
    args = parser.parse_args()

    input_indexes = matching_device_indexes("input", args.input)
    output_indexes = matching_device_indexes("output", args.output)
    print_device_list("input", input_indexes)
    print_device_list("output", output_indexes)

    if not input_indexes or not output_indexes:
        print("\nPass --input and --output with device indexes or name fragments to check a full-duplex pair.")
        return

    input_index = input_indexes[0]
    output_index = output_indexes[0]
    print(f"\nSelected pair: input={input_index}, output={output_index}")
    print("Individual input float32 rates:", supported_rates_for_device(input_index, kind="input", dtype="float32"))
    print("Individual output float32 rates:", supported_rates_for_device(output_index, kind="output", dtype="float32"))
    print("Note: int24 is shown for PortAudio diagnostics; AIEQ's NumPy stream UI uses float32/int32/int16.")
    duplex = supported_duplex_settings(input_index, output_index)
    if not duplex:
        print("Pair-compatible rates: no tested formats accepted by both devices.")
        return
    print("Pair-compatible rates accepted by both devices:")
    by_rate: dict[int, list[str]] = {}
    for rate, dtype in duplex:
        by_rate.setdefault(rate, []).append(dtype)
    for rate, dtypes in by_rate.items():
        print(f"  {rate}: {', '.join(dtypes)}")


if __name__ == "__main__":
    main()
