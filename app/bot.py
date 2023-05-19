import sys

sys.path.append("../app")
sys.path.append("..")  # for docker
from io import BytesIO

import librosa
import requests
import soundfile as sf
import yaml
from addict import Dict
from PIL import Image
from telegram import Update, Voice
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from src.utils import load_config, parse_args


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """audio handler"""
    voice = await update.message.voice.get_file()

    # voice_load_response = requests.get(photo.file_path)
    # image_bytes = BytesIO(image_load_response.content)

    response = requests.get(
        "http://ml.n19:30025/process_audio_url",
        files={"file": voice.file_id},
    )

    await update.message.reply_text(
        str(response.status_code), reply_to_message_id=update.message.id
    )


def main() -> None:
    config = load_config(parse_args())

    application = Application.builder().token(config.token).build()

    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.run_polling()


if __name__ == "__main__":
    main()
