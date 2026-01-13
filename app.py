# # # FILE: app.py
# # import streamlit as st
# # import os
# # import cv2
# # import torch
# # import torchaudio
# # import google.generativeai as genai
# # from PIL import Image
# # from transformers import AutoProcessor, AutoModelForCausalLM
# # from TTS.tts.configs.xtts_config import XttsConfig
# # from TTS.tts.models.xtts import Xtts
# # from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# # import numpy as np
# # import tempfile

# # # --- CONFIGURATION ---
# # st.set_page_config(page_title="AI Video Narrator", layout="wide")
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # --- CACHED FUNCTIONS (To load models only once) ---

# # @st.cache_resource
# # def load_git_model():
# #     """Loads Microsoft GIT for Image Captioning"""
# #     processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# #     model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(DEVICE)
# #     return processor, model

# # @st.cache_resource
# # def load_xtts_model():
# #     """Loads Coqui XTTS v2 manually (Bypassing Download Manager)"""
# #     base_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
# #     checkpoint_path = os.path.join(base_path, "model.pth")
# #     config_path = os.path.join(base_path, "config.json")
# #     vocab_path = os.path.join(base_path, "vocab.json")

# #     if not os.path.exists(checkpoint_path):
# #         st.error(f"❌ Critical Error: Could not find model at {checkpoint_path}")
# #         return None

# #     config = XttsConfig()
# #     config.load_json(config_path)
# #     model = Xtts.init_from_config(config)
    
# #     # Manual Weight Injection
# #     state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
# #     if "model" in state_dict: state_dict = state_dict["model"]
    
# #     # Clean keys
# #     keys_to_ignore = ["dvae", "torch_mel_spectrogram_style_encoder"]
# #     cleaned_state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in keys_to_ignore)}
    
# #     model.load_state_dict(cleaned_state_dict, strict=False)
# #     model.cuda()
    
# #     # Initialize Inference
# #     model.gpt.init_gpt_for_inference(kv_cache=True)
# #     model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    
# #     return model

# # # --- HELPER FUNCTIONS ---

# # def extract_frames(video_path, num_frames=5):
# #     """Extracts N evenly spaced frames from the video"""
# #     vidcap = cv2.VideoCapture(video_path)
# #     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     step = max(1, total_frames // num_frames)
    
# #     frames = []
# #     for i in range(0, total_frames, step):
# #         vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
# #         success, image = vidcap.read()
# #         if success:
# #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #             frames.append(Image.fromarray(image))
# #         if len(frames) >= num_frames:
# #             break
# #     vidcap.release()
# #     return frames

# # def generate_captions(frames, processor, model):
# #     """Generates captions for a list of images"""
# #     captions = []
# #     for frame in frames:
# #         inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
# #         generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
# #         caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# #         captions.append(caption)
# #     return captions

# # def generate_story(captions, api_key):
# #     """Uses Gemini to weave captions into a story"""
# #     genai.configure(api_key=api_key)
# #     model = genai.GenerativeModel('gemini-pro')
    
# #     prompt = f"""
# #     You are a narrator. I will give you a list of descriptions from a video. 
# #     Write a cohesive, short, and engaging story (max 3 sentences) connecting these scenes.
# #     Do not use "The video shows" or "In this scene". Just tell the story.
    
# #     Scenes:
# #     {captions}
# #     """
# #     response = model.generate_content(prompt)
# #     return response.text

# # # --- MAIN UI ---

# # st.title("🎬 AI Video Narrator (Clone Edition)")
# # st.sidebar.header("Configuration")

# # # 1. API Key Input
# # api_key = st.sidebar.text_input("Gemini API Key", type="password")

# # # 2. File Uploads
# # video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
# # voice_sample = st.file_uploader("Upload Your Voice Sample (.wav)", type=["wav"])

# # if st.button("🚀 Generate Narration"):
# #     if not api_key or not video_file or not voice_sample:
# #         st.error("Please provide an API Key, a Video, and a Voice Sample!")
# #     else:
# #         # A. SETUP
# #         status = st.empty()
# #         status.info("⏳ Loading AI Models... (This happens once)")
        
# #         git_processor, git_model = load_git_model()
# #         xtts_model = load_xtts_model()
        
# #         if not xtts_model:
# #             st.stop()

# #         # B. SAVE FILES
# #         tfile = tempfile.NamedTemporaryFile(delete=False)
# #         tfile.write(video_file.read())
# #         video_path = tfile.name

# #         vfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
# #         vfile.write(voice_sample.read())
# #         voice_path = vfile.name

# #         # C. VISION (See)
# #         status.info("👀 Watching video and extracting frames...")
# #         frames = extract_frames(video_path)
# #         st.image(frames, caption=[f"Frame {i+1}" for i in range(len(frames))], width=150)
        
# #         captions = generate_captions(frames, git_processor, git_model)
# #         with st.expander("See Raw Captions"):
# #             st.write(captions)

# #         # D. BRAIN (Think)
# #         status.info("🧠 Writing the script...")
# #         script = generate_story(captions, api_key)
# #         st.subheader("📝 Generated Script")
# #         st.write(f"*{script}*")

# #         # E. VOICE (Speak)
# #         status.info("🎙️ Cloning your voice and narrating...")
        
# #         # XTTS Inference
# #         gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[voice_path])
# #         out = xtts_model.inference(
# #             text=script,
# #             language="en",
# #             gpt_cond_latent=gpt_cond_latent,
# #             speaker_embedding=speaker_embedding,
# #             temperature=0.7,
# #         )
        
# #         # Save Audio
# #         output_audio_path = "final_narration.wav"
# #         torchaudio.save(output_audio_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
        
# #         status.success("🎉 Done!")
        
# #         # F. DISPLAY RESULTS
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.video(video_file)
# #         with col2:
# #             st.audio(output_audio_path)
# #             st.success("Play the audio to hear your AI Twin narrate the video!")




























# # update 1

# # FILE: app.py
# import streamlit as st
# import os
# import cv2
# import torch
# import torchaudio
# from openai import OpenAI  # <--- CHANGED: Using OpenRouter now
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# import numpy as np
# import tempfile

# # --- CONFIGURATION ---
# st.set_page_config(page_title="AI Video Narrator", layout="wide")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- CACHED FUNCTIONS ---

# @st.cache_resource
# def load_git_model():
#     """Loads Microsoft GIT for Image Captioning"""
#     # Using the video-fine-tuned model we verified works best
#     MODEL_NAME = "microsoft/git-base-vatex"
#     processor = AutoProcessor.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
#     return processor, model

# @st.cache_resource
# def load_xtts_model():
#     """Loads Coqui XTTS v2 manually (Bypassing Download Manager)"""
#     base_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
#     checkpoint_path = os.path.join(base_path, "model.pth")
#     config_path = os.path.join(base_path, "config.json")
#     vocab_path = os.path.join(base_path, "vocab.json")

#     if not os.path.exists(checkpoint_path):
#         st.error(f"❌ Critical Error: Could not find model at {checkpoint_path}")
#         return None

#     config = XttsConfig()
#     config.load_json(config_path)
#     model = Xtts.init_from_config(config)
    
#     # Manual Weight Injection
#     state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     if "model" in state_dict: state_dict = state_dict["model"]
    
#     keys_to_ignore = ["dvae", "torch_mel_spectrogram_style_encoder"]
#     cleaned_state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in keys_to_ignore)}
    
#     model.load_state_dict(cleaned_state_dict, strict=False)
#     model.cuda()
    
#     model.gpt.init_gpt_for_inference(kv_cache=True)
#     model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    
#     return model

# # --- HELPER FUNCTIONS ---

# def extract_frames(video_path, num_frames=6):
#     """Extracts 6 evenly spaced frames"""
#     vidcap = cv2.VideoCapture(video_path)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     frames = []
#     if total_frames > 0:
#         indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
#         for i in range(total_frames):
#             ret, frame = vidcap.read()
#             if not ret: break
#             if i in indices:
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(Image.fromarray(image))
#     vidcap.release()
#     return frames

# def generate_captions(frames, processor, model):
#     """Generates visual facts using GIT"""
#     # GIT expects a batch of images for video captioning
#     inputs = processor(images=[frames], return_tensors="pt")
#     pixel_values = inputs.pixel_values.to(DEVICE)
    
#     generated_ids = model.generate(pixel_values=pixel_values, max_length=50, num_beams=4)
#     caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return caption

# def generate_story(visual_fact, api_key):
#     """Uses OpenRouter (Llama 3) to style the caption"""
#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api_key,
#     )
    
#     prompt = f"""
#     You are an energetic video narrator.
#     VISUAL FACT: "{visual_fact}"
    
#     Task: Turn this boring fact into a short, engaging, spoken narration sentence (Max 15 words).
#     Make it sound like a live commentary.
#     """
    
#     completion = client.chat.completions.create(
#       model="meta-llama/llama-3.2-3b-instruct:free",
#       messages=[
#         {"role": "user", "content": prompt}
#       ]
#     )
#     return completion.choices[0].message.content.replace('"', '')

# # --- MAIN UI ---

# st.title("🎬 AI Video Narrator (Clone Edition)")
# st.sidebar.header("Configuration")

# # 1. API Key Input
# api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

# # 2. File Uploads
# video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
# voice_sample = st.file_uploader("Upload Your Voice Sample (.wav)", type=["wav"])

# if st.button("🚀 Generate Narration"):
#     if not api_key or not video_file or not voice_sample:
#         st.error("Please provide an API Key, a Video, and a Voice Sample!")
#     else:
#         # A. SETUP
#         status = st.empty()
#         status.info("⏳ Loading AI Models... (This happens once)")
        
#         git_processor, git_model = load_git_model()
#         xtts_model = load_xtts_model()
        
#         if not xtts_model:
#             st.stop()

#         # B. SAVE FILES
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(video_file.read())
#         video_path = tfile.name

#         vfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         vfile.write(voice_sample.read())
#         voice_path = vfile.name

#         # C. VISION (See)
#         status.info("👀 Watching video...")
#         frames = extract_frames(video_path)
#         st.image(frames[len(frames)//2], caption="AI View", width=300)
        
#         visual_fact = generate_captions(frames, git_processor, git_model)
#         with st.expander("See Raw Visual Fact"):
#             st.write(f"GIT Model saw: *{visual_fact}*")

#         # D. BRAIN (Think)
#         status.info("🧠 Writing the script...")
#         try:
#             script = generate_story(visual_fact, api_key)
#             st.subheader("📝 Generated Script")
#             st.write(f"*{script}*")

#             # E. VOICE (Speak)
#             status.info("🎙️ Cloning your voice...")
            
#             # XTTS Inference
#             gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[voice_path])
#             out = xtts_model.inference(
#                 text=script,
#                 language="en",
#                 gpt_cond_latent=gpt_cond_latent,
#                 speaker_embedding=speaker_embedding,
#                 temperature=0.7,
#             )
            
#             # Save Audio
#             output_audio_path = "final_narration.wav"
#             torchaudio.save(output_audio_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            
#             status.success("🎉 Done!")
            
#             # F. DISPLAY RESULTS
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.video(video_file)
#             with col2:
#                 st.audio(output_audio_path)
        
#         except Exception as e:
#             st.error(f"An error occurred: {e}")


























# update 2

# FILE: app.py
import streamlit as st
import os
import cv2
import torch
import torchaudio
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from moviepy.editor import VideoFileClip, AudioFileClip # <--- NEW IMPORT
import numpy as np
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Video Narrator", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CACHED FUNCTIONS ---

@st.cache_resource
def load_git_model():
    """Loads Microsoft GIT for Image Captioning"""
    MODEL_NAME = "microsoft/git-base-vatex"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return processor, model

@st.cache_resource
def load_xtts_model():
    """Loads Coqui XTTS v2 manually"""
    base_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
    checkpoint_path = os.path.join(base_path, "model.pth")
    config_path = os.path.join(base_path, "config.json")
    vocab_path = os.path.join(base_path, "vocab.json")

    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Critical Error: Could not find model at {checkpoint_path}")
        return None

    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    
    # Manual Weight Injection
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "model" in state_dict: state_dict = state_dict["model"]
    
    keys_to_ignore = ["dvae", "torch_mel_spectrogram_style_encoder"]
    cleaned_state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in keys_to_ignore)}
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.cuda()
    model.gpt.init_gpt_for_inference(kv_cache=True)
    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    return model

# --- HELPER FUNCTIONS ---

def extract_frames(video_path, num_frames=6):
    """Extracts 6 evenly spaced frames"""
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        for i in range(total_frames):
            ret, frame = vidcap.read()
            if not ret: break
            if i in indices:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(image))
    vidcap.release()
    return frames

def generate_captions(frames, processor, model):
    """Generates visual facts using GIT"""
    inputs = processor(images=[frames], return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50, num_beams=4)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

def generate_story(visual_fact, api_key):
    """Uses OpenRouter (Llama 3) to style the caption"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    prompt = f"""
    You are an energetic video narrator.
    VISUAL FACT: "{visual_fact}"
    Task: Turn this boring fact into a short, engaging, spoken narration sentence (Max 15 words).
    Make it sound like a live commentary.
    """
    completion = client.chat.completions.create(
      model="meta-llama/llama-3.2-3b-instruct:free",
      messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.replace('"', '')

def merge_video_audio(video_path, audio_path, output_path):
    """Merges the new audio into the original video"""
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # Trim audio if it's longer than video, or loop video? 
    # For now, we keep video length and let audio cut if too long, or silence if too short.
    # Ideally, we set audio and keep video duration.
    final_clip = video_clip.set_audio(audio_clip)
    
    # Write output
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# --- MAIN UI ---

st.title("🎬 AI Video Narrator (Ultimate Edition)")
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
voice_sample = st.file_uploader("Upload Your Voice Sample (.wav)", type=["wav"])

if st.button("🚀 Generate Narration"):
    if not api_key or not video_file or not voice_sample:
        st.error("Please provide an API Key, a Video, and a Voice Sample!")
    else:
        # A. SETUP
        status = st.empty()
        status.info("⏳ Loading AI Models...")
        git_processor, git_model = load_git_model()
        xtts_model = load_xtts_model()
        if not xtts_model: st.stop()

        # B. SAVE INPUTS
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        vfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        vfile.write(voice_sample.read())
        voice_path = vfile.name

        # C. VISION
        status.info("👀 Watching video...")
        frames = extract_frames(video_path)
        st.image(frames[len(frames)//2], caption="AI View", width=300)
        visual_fact = generate_captions(frames, git_processor, git_model)

        # D. BRAIN
        status.info("🧠 Writing script...")
        try:
            script = generate_story(visual_fact, api_key)
            st.subheader("📝 Generated Script")
            st.write(f"*{script}*")

            # E. VOICE
            status.info("🎙️ Cloning voice...")
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[voice_path])
            out = xtts_model.inference(
                text=script,
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
            )
            output_audio_path = "temp_audio.wav"
            torchaudio.save(output_audio_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

            # F. MERGE (The New Part)
            status.info("🎬 Merging video and audio...")
            final_output = "final_narrated_video.mp4"
            merge_video_audio(video_path, output_audio_path, final_output)
            
            status.success("🎉 DONE! Watch your creation below.")
            
            # G. DISPLAY & DOWNLOAD
            st.video(final_output)
            
            with open(final_output, "rb") as file:
                st.download_button(
                    label="⬇️ Download Video",
                    data=file,
                    file_name="my_ai_video.mp4",
                    mime="video/mp4"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")