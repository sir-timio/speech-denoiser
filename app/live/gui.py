import os
import sys

sys.path.append("../app")

import tkinter as tk
from tkinter import HORIZONTAL, Entry, IntVar, Label, OptionMenu, Scale, StringVar
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter.ttk import Label, Style

import sounddevice as sd
import torch
from src.models import load_denoiser

from live.streamer import DemucsStreamer
from live.utils import parse_audio_device, query_devices

W, H = 720, 480
W_BIAS = 0
BG = "#ce42f5"
BTN_BG = "green"
LBL_BG = "white"
TITLE = "Real-time denoiser"


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame = None
        self.title_font = tkfont.Font(
            family="Helvetica", size=18, weight="bold", slant="italic"
        )
        self.eval("tk::PlaceWindow . center")
        self.geometry(f"{W}x{H}-{W_BIAS}+0")
        self.configure(background=BG)
        self.title(TITLE)
        self.focus()
        self.switch_frame(MainPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack_propagate(0)
        self._frame.pack()


##################################################
#                                                #
#               Main Page                        #
#                                                #
##################################################


class MainPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.style = Style()
        self.style.theme_use("alt")
        self.running = False

        self.set_drop_input(
            ["out", "ckpt", "num_threads"],
            ["BlackHole 2ch", "demucs_48.ckpt", "1"],
            [
                [d["name"] for d in sd.query_devices()],
                os.listdir("weights"),
                list(range(16)),
            ],
            labels=[
                "Выберите устройство",
                "Выберите модель",
                "Выберите кол-во потоков",
            ],
            row=0,
            col=0,
            sticky="e",
        )

        Label(self, text=f"Введите размер пакета: ").grid(row=4, column=0, sticky="e")
        self.num_frames = Entry(self, width=10)
        self.num_frames.insert(0, "1")
        self.num_frames.grid(row=4, column=1, sticky="e")

        self.dry = IntVar()
        self.dry_label = Label(self)
        self.dry.trace("w", self.format_scale)
        self.dry.set(4)
        self.dry_label.grid(row=5, column=0, pady=30, padx=30)
        Scale(
            self,
            orient=HORIZONTAL,
            from_=0,
            to=100,
            command=self.onScale,
            length=200,
            variable=self.dry,
        ).grid(row=5, column=1)

        save_btn = tk.Button(self, text="Сохранить")
        save_btn.config(command=self.init_streamer)
        save_btn.grid(row=6, column=1)

        start_btn = tk.Button(self, text="Старт")
        start_btn.config(command=self.start)
        start_btn.grid(row=7, column=0)

        stop_btn = tk.Button(self, text="Стоп")
        stop_btn.config(command=self.stop)
        stop_btn.grid(row=7, column=1)

    def init_stream_in(self, sample_rate):
        device_in = parse_audio_device(None)
        caps = query_devices(device_in, "input")
        channels_in = min(caps["max_input_channels"], 2)
        self.stream_in = sd.InputStream(
            device=device_in, samplerate=sample_rate, channels=channels_in
        )

    def init_stream_out(self, sample_rate):
        device_out = parse_audio_device(self.out.get())
        caps = query_devices(device_out, "output")
        self.channels_out = min(caps["max_output_channels"], 2)
        self.stream_out = sd.OutputStream(
            device=device_out, samplerate=sample_rate, channels=self.channels_out
        )

    def init_streamer(self):
        try:
            torch.set_num_threads(int(self.num_threads.get()))

            model = load_denoiser(ckpt_path=f"weights/{self.ckpt.get()}")
            model.eval()
            print("Model loaded.")
            self.model = model
            self.streamer = DemucsStreamer(
                model, dry=self.dry.get(), num_frames=int(self.num_frames.get())
            )
            self.init_stream_in(model.sample_rate)
            self.init_stream_out(model.sample_rate)
            self.stream_in.start()
            self.stream_out.start()
        except Exception as e:
            print(e)
            messagebox.showerror("Ошибка", e)

    def start(self):
        self.running = True
        self.after(1000, self.denoise)

    def stop(self):
        self.running = False

    def denoise(self):
        if not self.running:
            return
        first = True
        current_time = 0
        last_log_time = 0
        last_error_time = 0
        cooldown_time = 2
        log_delta = 10
        sr_ms = self.model.sample_rate / 1000
        stride_ms = self.streamer.stride / sr_ms
        print(
            f"Ready to process audio, total lag: {self.streamer.total_length / sr_ms:.1f}ms."
        )
        while self.running:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = self.streamer.time_per_frame * 1000
                rtf = tpf / stride_ms
                print(f"time per frame: {tpf:.1f}ms, ", end="")
                print(f"RTF: {rtf:.1f}")
                self.streamer.reset_time_per_frame()

            length = self.streamer.total_length if first else self.streamer.stride
            first = False
            current_time += length / self.model.sample_rate
            frame, overflow = self.stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1)
            with torch.no_grad():
                out = self.streamer.feed(frame[None])[0]
            if not out.numel():
                continue
            out = out[:, None].repeat(1, self.channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = self.stream_out.write(out)
            if overflow or underflow:
                if current_time >= last_error_time + cooldown_time:
                    last_error_time = current_time
                    tpf = 1000 * self.streamer.time_per_frame
                    print(
                        f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
                        f"(should be less than {stride_ms:.1f}ms)."
                    )

    def format_scale(self, a, b, c):
        self.dry_label["text"] = "Удаление шума {}%".format(self.dry.get())

    def onScale(self, v):
        self.dry.set(float(v))

    def set_drop_input(
        self, fields, names, options, labels=None, row=0, col=0, sticky="e"
    ):
        if labels is None:
            labels = [""] * len(names)
        for i, t in enumerate(zip(fields, names, options, labels)):
            field, name, ops, label = t
            self.__dict__[field] = StringVar()
            drop = OptionMenu(self, self.__dict__[field], *ops)
            self.__dict__[field].set(name)
            Label(self, text=f"{label}:").grid(row=row + i, column=col, sticky="E")
            drop.grid(row=row + i, column=col + 1, sticky=sticky)


if __name__ == "__main__":
    app = App()
    app.mainloop()
