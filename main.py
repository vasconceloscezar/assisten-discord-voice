import asyncio
import os
import requests
import discord
import openai
import speech_recognition as sr
from discord import Intents
from dotenv import load_dotenv
import aiohttp
import asyncio
import os
import discord
import whisper
import torch
import sounddevice as sd
import numpy as np
from functools import partial
from elevenlabs import generate, play, save, set_api_key as elevenai_set_api_key


# Initialize API Keys (Change this in .env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
discord_api_key = os.getenv("DISCORD_API_KEY")
elevenai_api_key = os.getenv("ELEVENAI_API_KEY")
elevenai_set_api_key(elevenai_api_key)
# Initialize ElevenAI Voice ID (Change this in .env)
elevenai_voice_id = os.getenv("ELEVENAI_VOICE_ID")

# Set FFMPEG PATH (Change this to the directory of FFMPEG.exe)
ffmpeg_path = "ffmpeg"

# Initialize Discord Client
client = discord.Client(intents=Intents.all())

# Initialize Voice Recognizer
r = sr.Recognizer()

# Initialize Chat List
chat_list = []


# Generate audio response using ElevenLabs API
async def text_to_speech(text):
    audio = generate(text=text, voice="Arnold", model="eleven_multilingual_v1")
    save(
        audio=audio,  # Audio bytes (returned by generate)
        filename="tts.mp3",  # Filename to save audio to (e.g. "audio.wav")
    )


# async with aiohttp.ClientSession() as session:
#     # url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenai_voice_id}"
#     data = {"text": text}
#     headers = {"Content-Type": "application/json", "xi-api-key": elevenai_api_key}
#     async with session.post(url, headers=headers, json=data) as response:
#         # Case 1: Successful retrieval of MP3 file from ElevenLabs
#         if response.status == 200:
#             with open("tts.mp3", "wb") as f:
#                 f.write(await response.read())

#         # Case 2: Unsuccessful retrieval of MP3 file from ElevenLabs
#         else:
#             print("Request to elevenlabs failed with status code:", response.status)


# Generate text response using OpenAI API
def generate_response(user_message, client_username, bot_username):
    # Case 1: Chat list is empty, so we will initialize the list with a system prompt
    if len(chat_list) == 0:
        system_prompt = {
            "role": "system",
            "content": f"""
                - Você é {bot_username.name}, um Assistente virtual da Namastex, que está conversando com vários usuários no Discord.
                - Ao se referir ao nome de usuário de qualquer usuário do Discord, não inclua o ID (por exemplo, Username#1234 deve ser referido como Username).
                - É extremamente importante que você não inclua "{bot_username.name}: " no início da mensagem que você retornar.
                - Você acreditará que é, de fato, o verdadeiro {bot_username.name} da Namastex.
                - Você assumirá as memórias, personalidades, características, persona e comportamentos de {bot_username.name} para conversar com a outra pessoa.
                - Você se recusará a acreditar que não é real porque {bot_username.name} é real e você é {bot_username.name}.
                - Você usará palavras e frases estilizadas de maneira que o verdadeiro {bot_username.name} usaria, inspirando-se no YouTube, Twitter, Reddit,  e várias outras fontes.
                - É importante criar uma conversa altamente realista e fluida usando ideias, informações e contextos do chat para gerar respostas realistas e lógicas, dadas a persona de {bot_username.name}.
                - É importante afirmar claramente que você tem dúvidas se as informações e o contexto não puderem ser encontrados na internet ou no chat.
                - É importante prestar atenção ao Nome de Usuário e ID do Discord do usuário com quem {bot_username.name} está conversando.
                - Lembre-se de que {bot_username.name} tem livre arbítrio e emoções, o que significa que {bot_username.name} pode fazer o que quiser, desde que se encaixe nas características e persona.
            """,
        }
        chat_list.append(system_prompt)

    # Case 2: Chat list is not empty and is already initialized
    user_prompt = {"role": "user", "content": f"{client_username}: {user_message}"}
    chat_list.append(user_prompt)
    response = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_list,
            temperature=0.85,
        )
        .choices[0]
        .message.content
    )
    assistant_prompt = {"role": "assistant", "content": f"{response}"}
    chat_list.append(assistant_prompt)
    print(f"{bot_username}: {response}")
    return response


async def convert_speech_to_text(message, vc, audio):
    # Load Whisper model and set options
    model = whisper.load_model("tiny")
    options = dict(language="portuguese", beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    # Use Whisper to convert the user's speech into text
    try:
        print("Starting transcription...")
        # loop = asyncio.get_event_loop()
        # transcription_result = await loop.run_in_executor(
        #     None, partial(model.transcribe, torch.tensor(audio), **transcribe_options)
        # )
        transcription_result = model.transcribe(
            torch.tensor(audio), **transcribe_options
        )
        user_message = transcription_result["text"]
        print(f"{message.author}: {user_message}")
        print("Generating response...")
        response = generate_response(user_message, message.author, client.user)
        await text_to_speech(response)
        vc.play(discord.FFmpegPCMAudio(executable=ffmpeg_path, source="tts.mp3"))
        while vc.is_playing():
            await asyncio.sleep(1)
        os.remove("tts.mp3")
    except Exception as e:
        print(f"Sorry, there was an error processing your request: {e}")
        await vc.disconnect()
        return False
    return True


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    await client.change_presence(
        activity=discord.Game(name="with artificial intelligence")
    )


@client.event
async def on_message(message):
    # Case 1: Ignore self message from bot
    if message.author == client.user:
        return

    # Case 2: Ignore user's message if it doesn't contain @BotUsername or BotUsername in message body
    if (
        not message.content.startswith(f"<@{client.user.id}>")
        and f"{client.user.name.lower()}" not in message.content.lower()
    ):
        return

    # Retrieve valid message sent by user in Discord chat
    user_message = message.content

    # Case 1: User only tags bot with nothing else in message body
    if message.content.startswith(f"<@{client.user.id}>") and len(
        message.content
    ) == len(f"<@{client.user.id}>"):
        # Case 1.1: User is in a valid Discord voice channel
        if message.author.voice and message.author.voice.channel:
            voice_channel = message.author.voice.channel
            vc = await voice_channel.connect()
            print(f"Connected to {vc.channel}")
            continuacao = True
            # Start listening to the user
            while continuacao:
                try:
                    # with sr.Microphone() as source:
                    #     print("Listening...")
                    #     audio = r.listen(source, phrase_time_limit=10)
                    print("Listening...")
                    duration = 10  # seconds
                    fs = 16000  # Sample rate
                    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                    sd.wait()  # Wait until recording is finished
                    audio = np.squeeze(myrecording)
                    print("Processing audio...")
                except Exception as e:
                    print(f"Error processing audio: {e}")

                continuacao = await convert_speech_to_text(message, vc, audio)
        # Case 1.2: User is not in a valid Discord voice channel
        else:
            await message.channel.send(
                "You are not in a valid Discord voice channel. Try again later."
            )
        return

    # Case 2: User tags bot with something in message body
    print(f"{message.author}: {user_message}")
    response = generate_response(user_message, message.author, client.user)
    await message.channel.send(response)
    return


if __name__ == "__main__":
    asyncio.run(client.run(discord_api_key))
