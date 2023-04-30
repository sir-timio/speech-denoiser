import sys

sys.path.append("../app")
import sounddevice as sd
import torch
from src.utils import load_config, parse_args
from live.streamer import DemucsStreamer
from src.models import load_denoiser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = f"Invalid {kind} audio interface {device}.\n"
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this."
        )
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def main():
    args = load_config(parse_args())
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    model = load_denoiser(ckpt_path=args.denoiser_ckpt_path)
    model.eval()
    print("Model loaded.")
    streamer = DemucsStreamer(model, dry=args.dry, num_frames=args.num_frames)

    device_in = parse_audio_device(args.in_)
    caps = query_devices(device_in, "input")
    channels_in = min(caps["max_input_channels"], 2)
    stream_in = sd.InputStream(
        device=device_in, samplerate=model.sample_rate, channels=channels_in
    )

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps["max_output_channels"], 2)
    stream_out = sd.OutputStream(
        device=device_out, samplerate=model.sample_rate, channels=channels_out
    )

    stream_in.start()
    stream_out.start()
    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = model.sample_rate / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    while True:
        try:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = streamer.time_per_frame * 1000
                rtf = tpf / stride_ms
                print(f"time per frame: {tpf:.1f}ms, ", end="")
                print(f"RTF: {rtf:.1f}")
                streamer.reset_time_per_frame()

            length = streamer.total_length if first else streamer.stride
            first = False
            current_time += length / model.sample_rate
            frame, overflow = stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1).to(args.device)
            with torch.no_grad():
                out = streamer.feed(frame[None])[0]
            if not out.numel():
                continue
            if args.compressor:
                out = 0.99 * torch.tanh(out)
            out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = stream_out.write(out)
            if overflow or underflow:
                if current_time >= last_error_time + cooldown_time:
                    last_error_time = current_time
                    tpf = 1000 * streamer.time_per_frame
                    print(
                        f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
                        f"(should be less than {stride_ms:.1f}ms)."
                    )
        except KeyboardInterrupt:
            print("Stopping")
            break
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
