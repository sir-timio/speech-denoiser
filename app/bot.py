import sys

sys.path.append("../app")
sys.path.append("..")  # for docker
import base64
import io

import aiohttp
import librosa
import requests
import soundfile as sf
import yaml
from addict import Dict
from PIL import Image
from telegram import InputFile, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from src.utils import load_config, parse_args


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """audio handler"""
    voice = await update.message.voice.get_file()

    # Download the voice file
    async with aiohttp.ClientSession() as session:
        async with session.get(voice.file_path) as resp:
            voice_bytes = await resp.read()

    # Send the file to the server
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("file", io.BytesIO(voice_bytes), filename="voice.ogg")
        async with session.post(
            "http://localhost:30025/process_audio", data=data
        ) as resp:
            if resp.status == 200:
                body = await resp.json()
                denoised_voice = InputFile(
                    io.BytesIO(base64.b64decode(body["voice"])),
                    filename="denoised_voice.ogg",
                )
                await update.message.reply_voice(
                    denoised_voice,
                    caption=f"{str(resp.status)}: {body['text']}",
                    reply_to_message_id=update.message.id,
                )
            else:
                await update.message.reply_text(
                    f"{resp.status}: {resp.reason}",
                    reply_to_message_id=update.message.id,
                )


def main() -> None:
    config = load_config(parse_args())

    application = Application.builder().token(config.token).build()

    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.run_polling()


if __name__ == "__main__":
    main()
