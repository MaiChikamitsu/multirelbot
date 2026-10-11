import argparse
import os
import time
import wave
from io import BytesIO

os.environ.setdefault("MULTIRELBOT_SKIP_SOCKETIO", "1")


def list_input_devices():
    import pyaudio

    p = pyaudio.PyAudio()
    try:
        print("Input devices:")
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                name = info.get("name", "")
                channels = int(info.get("maxInputChannels", 0))
                rate = int(info.get("defaultSampleRate", 0))
                print(f"  {index}: {name} ({channels}ch, {rate}Hz)")
    finally:
        p.terminate()


def build_wav_buffer(frames, rc):
    audio_buffer = BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(rc.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rc.SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    audio_buffer.seek(0)
    return audio_buffer


def process_frames(frames, rc, min_bytes):
    audio_buffer = build_wav_buffer(frames, rc)
    byte_count = audio_buffer.getbuffer().nbytes
    duration = len(b"".join(frames)) / (rc.SAMPLE_RATE * 2)

    print(f"\n--- utterance: {duration:.1f}s / {byte_count} bytes ---")
    if byte_count < min_bytes:
        print(f"skip: 音声が短いので判定しません ({byte_count} < {min_bytes})")
        return None

    speaker = rc.identify_speaker(audio_buffer)
    print(f"result: {speaker}")
    return speaker


def register_config_speakers(rc):
    rc.known_speakers.clear()
    speakers = getattr(rc._CFG.participants, "speakers", None) or {}
    if not speakers:
        raise RuntimeError("config.local.yaml の participants.speakers が空です。")

    for speaker_name, audio_path in speakers.items():
        rc.register_reference_speaker(speaker_name, audio_path)
        print(f"registered: {speaker_name} <- {audio_path}")


def run(args):
    import pyaudio
    import realtime_communicator as rc

    register_config_speakers(rc)

    silence_duration = (
        args.silence_duration
        if args.silence_duration is not None
        else rc.SILENCE_DURATION
    )
    min_bytes = (
        args.min_bytes if args.min_bytes is not None else rc.SKIP_THRESHOLD_BYTES
    )

    p = pyaudio.PyAudio()
    stream_kwargs = {
        "format": rc.FORMAT,
        "channels": rc.CHANNELS,
        "rate": rc.SAMPLE_RATE,
        "input": True,
        "frames_per_buffer": rc.CHUNK,
    }
    if args.input_device_index is not None:
        stream_kwargs["input_device_index"] = args.input_device_index

    stream = p.open(**stream_kwargs)
    print("\nSpeaker recognition test started.")
    print("Speak, then pause. Press Ctrl+C to stop.")
    print(f"silence_duration={silence_duration}, min_bytes={min_bytes}")

    frames = []
    silence_start_time = None
    has_voice = False

    try:
        while True:
            data = stream.read(rc.CHUNK, exception_on_overflow=False)
            frames.append(data)

            if rc.is_speech(data, rc.SAMPLE_RATE):
                has_voice = True
                silence_start_time = None
                continue

            if silence_start_time is None:
                silence_start_time = time.time()
                continue

            if has_voice and time.time() - silence_start_time >= silence_duration:
                process_frames(frames, rc, min_bytes)
                if args.once:
                    break
                frames = []
                has_voice = False
                silence_start_time = None
            elif not has_voice:
                frames = []
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Record from the microphone and run only speaker identification."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        help="PyAudio input device index. Omit to use the Mac's current default input.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Stop after one detected utterance.",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        help="Seconds of silence used to end one utterance.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        help="Minimum WAV byte size before speaker identification runs.",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    run(args)


if __name__ == "__main__":
    main()
