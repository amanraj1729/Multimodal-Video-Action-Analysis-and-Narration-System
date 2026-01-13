# 🎬 AI Video Narrator

A multimodal AI application that automatically generates audio narration for videos using a clone of the user's voice.

## 🚀 Features
- **Vision:** Uses Microsoft GIT (Generative Image-to-Text) to analyze video frames and understand the scene.
- **Brain:** Uses Llama 3 (via OpenRouter) to write an engaging, witty script based on the visual analysis.
- **Voice:** Uses Coqui XTTS v2 to clone the user's voice and narrate the script.
- **Production:** Merges the generated audio with the original video automatically using MoviePy.

## 🛠️ Tech Stack
- **Python** (Streamlit, PyTorch, OpenCV)
- **Models:** Microsoft GIT-base, Llama 3, XTTS v2
- **Deployment:** Hugging Face Spaces (CPU Basic)

## 📦 How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 🎥 Demo
[Link to your Hugging Face Space or a screenshot of the app running]