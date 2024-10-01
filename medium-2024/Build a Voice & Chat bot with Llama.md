# [Build a Voice & Chat bot with Llama](https://medium.com/@yuxiaojian/build-a-voice-chat-bot-with-llama-aa0abf8437f5)

Previously, I shared a story about building an  [OpenAI Voice Chat Bot](https://medium.com/@yuxiaojian/an-openai-voice-chat-bot-b4cbe553f3ca)  which worked quite well and provided a lot of fun for kids. However, the associated costs can be a concern.

Is it possible to use free libraries to replace the paid APIs without significantly compromising performance? The answer is yes, and it’s surprisingly straightforward to achieve using the  [OpenAI Voice Chat Bot](https://medium.com/@yuxiaojian/an-openai-voice-chat-bot-b4cbe553f3ca)  framework. The key is to replace the following components:

-   Large Language (LLM): Switch from OpenAI GPT-4 to Llama 3.1.
-   Speech to Text (STT): Replace OpenAI Whisper-1 with Python Whisper.
-   Text to Speech (TTS): Substitute OpenAI TTS with  [Google TTS](https://pypi.org/project/gTTS/)

If you haven’t read the  [OpenAI Voice Chat Bot](https://medium.com/@yuxiaojian/an-openai-voice-chat-bot-b4cbe553f3ca), I recommend checking it out first. This Llama bot is built on the same framework as the OpenAI bot, making the transition seamless.

# Setting up the Llama server

First thing first, we need to set up a Llama. First, follow these  [instructions](https://github.com/ollama/ollama)  to set up and run a local Ollama instance:

-   Download and install Ollama on your supported platform.
-   Fetch available LLM model via  `ollama pull <name-of-model>`, e.g.,  `ollama pull llama3.1`
-   This will download the default tagged version of the model.
-   To view all pulled models, use  `ollama list`
-   To chat directly with a model from the command line, use  `ollama run <name-of-model>`
-   Find the running model with  `ollama ps`
-   Run  `ollama help`  in the terminal to see available commands too.

By following these steps, you’ll have your Llama server up and running. Double-check if a process is listening on port 11434.
```
> lsof -i :11434  
COMMAND   PID      USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME  
ollama  15472 user   3u  IPv4 0xac6a351cfa11abcd      0t0  TCP localhost:11434 (LISTEN)
```
# Speech to Text (STT)

The Whisper model is loaded using  `@st.cache_resource`  to prevent reloading on each run.
```
# Function to transcribe audio  
@st.cache_resource  
def load_whisper_model(model_size):  
    return whisper.load_model(model_size)
```
You can select the Whisper model size from these options:  `["base", "tiny", "small", "medium", "large"]`.
```
def transcribe_audio(audio_file):  
  
    # Options for the transcribe model ["base", "tiny", "small", "medium", "large"]  
    model = load_whisper_model("base")  
    result = model.transcribe(audio_file, fp16=False)  
    return result["text"]
```
The application uses a temporary file to store the recorded audio before transcription. This temporary file is to ensure smooth operation and is deleted after use.

# If audio bytes are available, transcribe and add to chat history  
if audio_bytes is not None:  
```  
    # Save the recorded audio to a temporary file  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:  
        temp_audio.write(audio_bytes)  
        temp_audio_path = temp_audio.name  
  
    with st.spinner("Transcribing..."):  
        transcription = transcribe_audio(temp_audio_path)  
        ...  
  
    # Clean up  
    os.unlink(temp_audio_path)
```
# Text to Speech (TTS)

For text-to-speech functionality, I use gTTS (Google Text-to-Speech), which is both popular and reliable. The process involves generating an MP3 audio file using gTTS and then playing it with  `st.audio`.
```
# Function for text-to-speech  
def text_to_speech(text, lang='en'):  
    tts = gTTS(text=text, lang=lang)  
    mp3_fp = io.BytesIO()  
    tts.write_to_fp(mp3_fp)  
    return mp3_fp.getvalue()  
...  
tts_audio = text_to_speech(transcription)  
st.audio(tts_audio, format="audio/mpeg", autoplay=True)
```
Ensure the audio format is  `audio/mpeg`  instead of  `audio/mp3`  . The  `audio/mp3`  format may not play on mobile devices, whereas  `audio/mpeg`  is compatible with both mobile and laptop platforms.

# Putting It Together

We’ve covered all the key components. The full code can be found in the `llama_voice_chat_bot.py` file. You can run this on your laptop and access the website remotely. More detailed instructions are available in the `voice_chat_bot` repository.

We’ve covered all the key components. The full code can be found in the  [llama_voice_chat_bot.py](https://github.com/yuxiaojian/voice_chat_bot/blob/main/llama_voice_chat_bot.py)  file. You can run this on your laptop and access the website remotely. More detailed instructions are available in the  [voice_chat_bot](https://github.com/yuxiaojian/voice_chat_bot)  repository.

Now, you have everything you need to run the app — a fully functioning, LLM-powered voice and chat bot, all free of charge. Enjoy!