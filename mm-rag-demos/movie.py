import sys
import os
print("Python executable:", sys.executable)  # Should point to .venv\Scripts\python.exe
print("Python version:", sys.version)

try:
    from moviepy.editor import VideoFileClip
    print("MoviePy imported successfully!")
except ModuleNotFoundError as e:
    print(f"Failed to import moviepy.editor: {e}")
    print("Installed packages:", os.popen("pip list").read())
    sys.exit(1)

import pytube
import whisper
import clip
import torch
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from transformers import LlavaForConditionalGeneration
import gradio as gr

# Step 1: Download and preprocess video
video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"  # Example video
output_path = "./video_data"
os.makedirs(output_path, exist_ok=True)
yt = pytube.YouTube(video_url).streams.first().download(output_path=output_path)
video_path = os.path.join(output_path, "input_video.mp4")
video = VideoFileClip(video_path)
frames = [frame for frame in video.iter_frames(fps=0.2)]  # 1 frame every 5s
audio_path = os.path.join(output_path, "audio.wav")
video.audio.write_audiofile(audio_path)

try:
    model_whisper = whisper.load_model("base")
    transcript = model_whisper.transcribe(audio_path)["text"]
except Exception as e:
    print(f"Error with Whisper: {e}")
    transcript = "Fallback transcript due to Whisper failure."

# Step 2: Embed frames and text
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
frame_embeddings = [
    clip_model.encode_image(
        preprocess(torch.from_numpy(frame).permute(2, 0, 1)).unsqueeze(0).to(device)
    ) for frame in frames
]
text_embedding = clip_model.encode_text(clip.tokenize(transcript).to(device))

# Step 3: Store in LanceDB
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
index = MultiModalVectorStoreIndex.from_documents(
    documents=[{"text": transcript, "images": frames}], storage_context=storage_context
)

# Step 4: Query and generate (simplified for demo)
retriever = index.as_retriever(similarity_top_k=3)
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-13b")

def query_video(question):
    results = retriever.retrieve(question)
    context_text = [r.text for r in results if hasattr(r, 'text')]
    context_images = [r.image for r in results if hasattr(r, 'image')]
    response = f"Based on the video: {context_text[0] if context_text else 'No text found'}"
    return response, context_images[0] if context_images else None

# Step 5: Gradio interface
interface = gr.Interface(fn=query_video, inputs="text", outputs=["text", "image"])
interface.launch()