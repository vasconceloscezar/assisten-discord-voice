# Voice Assistant for Discord 

Huge thanks to ErnestAroozoo for [GPT-Discord-Chatbot](https://github.com/ErnestAroozoo/GPT-Discord-Chatbot)

Did some alterations to use Whisper for Speech to text and adjusted ElevenLabs API. 


## Setup 

First adjust your env variables following the [.env-example](./.env-example) file. 

Get your OpenAI API key [here]()
Get your ElevenLabs API key [here]()
Set up your Discord bot [here]()

For Elven Labs, we are going to use the voice name. Here's the list of voices premade provided by Elven Labs: 

| Name   |
|--------|
| Rachel |
| Domi   |
| Bella  |
| Antoni |
| Elli   |
| Josh   |
| Arnold |
| Adam   |
| Sam    |

Choose one and put in the `ELEVENAI_VOICE_ID` env var

Add the bot to your server.

## Running 

To run your discord bot, execute: 

```bash
python ./main.py
```

Join a Voice Channel and mention the bot in any chat message. 

The bot will join your voice chat and start listening. 
