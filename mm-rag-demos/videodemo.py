import pytube, clip, torch
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from transformers import LlavaForConditionalGeneration
import gradio as gr
from pathlib import Path
import os
from os import path as osp
import json
import cv2
import webvtt
import whisper
from moviepy.editor import VideoFileClip
from PIL import Image
import base64


# Step 1: Download and preprocess video
video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"  # Gaussian Function video
yt = pytube.YouTube(video_url).streams.first().download(output_path="./video_data")
video = moviepy.editor.VideoFileClip("./video_data/input_video.mp4")
frames = [frame for frame in video.iter_frames(fps=0.2)]  # Extract 1 frame every 5s
audio = video.audio.write_audiofile("./audio.wav")
model_whisper = whisper.load_model("base")
transcript = model_whisper.transcribe("./audio.wav")["text"]

# Step 2: Embed frames and text
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
frame_embeddings = [clip_model.encode_image(preprocess(frame).unsqueeze(0).to("cuda")) for frame in frames]
text_embedding = clip_model.encode_text(clip.tokenize(transcript).to("cuda"))

# Step 3: Store in LanceDB
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
index = MultiModalVectorStoreIndex.from_documents(
    documents=[{"text": transcript, "images": frames}], storage_context=storage_context
)

# Step 4: Query and generate
retriever = index.as_retriever(similarity_top_k=3)
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-13b")

def query_video(question):
    results = retriever.retrieve(question)
    context = [r.text for r in results] + [r.image for r in results]
    response = llava_model.generate(context, question)
    return response, context[1]  # Return text and first retrieved frame

# Step 5: Gradio interface
interface = gr.Interface(fn=query_video, inputs="text", outputs=["text", "image"])
interface.launch()