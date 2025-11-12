import gradio as gr
import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Document processing imports
import PyPDF2
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_openai.embeddings import AzureOpenAIEmbeddings

api_key = os.getenv("DIAL_API_KEY")

embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="text-embedding-ada-002",
    #model=azure_configs["embedding_name"],
)


class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.embedding_model = embeddings
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error processing text file {txt_path}: {e}")
            return ""
    
    def extract_text_from_image(self, img_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(img_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not chunks:
            return np.array([])
        return self.embedding_model.encode(chunks)

class DocumentStore:
    """Stores and manages processed documents"""
    
    def __init__(self, store_path: str = "document_store.pkl"):
        self.store_path = store_path
        self.documents = []
        self.embeddings = None
        self.load_store()
    
    def add_document(self, filepath: str, content: str, chunks: List[str], embeddings: np.ndarray):
        """Add a document to the store"""
        doc_info = {
            'filepath': filepath,
            'content': content,
            'chunks': chunks,
            'embeddings': embeddings
        }
        self.documents.append(doc_info)
        
        # Update combined embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.save_store()
    
    def search_similar(self, query: str, processor: DocumentProcessor, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks to the query"""
        if not self.documents or self.embeddings is None:
            return []
        
        query_embedding = processor.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        chunk_idx = 0
        
        for doc in self.documents:
            doc_chunks = len(doc['chunks'])
            for i in range(doc_chunks):
                if chunk_idx in top_indices:
                    results.append({
                        'filepath': doc['filepath'],
                        'chunk': doc['chunks'][i],
                        'similarity': similarities[chunk_idx]
                    })
                chunk_idx += 1
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def save_store(self):
        """Save document store to file"""
        try:
            with open(self.store_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
        except Exception as e:
            logger.error(f"Error saving document store: {e}")
    
    def load_store(self):
        """Load document store from file"""
        try:
            if os.path.exists(self.store_path):
                with open(self.store_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', None)
        except Exception as e:
            logger.error(f"Error loading document store: {e}")
            self.documents = []
            self.embeddings = None

class RAGTools:
    """Tools for the RAG agent"""
    
    def __init__(self, processor: DocumentProcessor, store: DocumentStore):
        self.processor = processor
        self.store = store
    
    def process_pdf_tool(self, filepath: str) -> str:
        """Tool: Process PDF file"""
        if not os.path.exists(filepath):
            return f"Error: PDF file '{filepath}' not found."
        
        content = self.processor.extract_text_from_pdf(filepath)
        if not content:
            return f"Error: Could not extract text from PDF '{filepath}'."
        
        chunks = self.processor.chunk_text(content)
        embeddings = self.processor.create_embeddings(chunks)
        
        self.store.add_document(filepath, content, chunks, embeddings)
        
        return f"Successfully processed PDF '{filepath}'. Extracted {len(chunks)} text chunks."
    
    def process_text_tool(self, filepath: str) -> str:
        """Tool: Process text file"""
        if not os.path.exists(filepath):
            return f"Error: Text file '{filepath}' not found."
        
        content = self.processor.extract_text_from_txt(filepath)
        if not content:
            return f"Error: Could not read text from '{filepath}'."
        
        chunks = self.processor.chunk_text(content)
        embeddings = self.processor.create_embeddings(chunks)
        
        self.store.add_document(filepath, content, chunks, embeddings)
        
        return f"Successfully processed text file '{filepath}'. Extracted {len(chunks)} text chunks."
    
    def process_image_tool(self, filepath: str) -> str:
        """Tool: Process image file"""
        if not os.path.exists(filepath):
            return f"Error: Image file '{filepath}' not found."
        
        content = self.processor.extract_text_from_image(filepath)
        if not content:
            return f"Warning: No text extracted from image '{filepath}'."
        
        chunks = self.processor.chunk_text(content)
        embeddings = self.processor.create_embeddings(chunks)
        
        self.store.add_document(filepath, content, chunks, embeddings)
        
        return f"Successfully processed image '{filepath}'. Extracted {len(chunks)} text chunks."
    
    def search_documents_tool(self, query: str) -> str:
        """Tool: Search through processed documents"""
        results = self.store.search_similar(query, self.processor, top_k=3)
        
        if not results:
            return "No relevant documents found."
        
        response = "Found relevant information:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. From '{os.path.basename(result['filepath'])}' (similarity: {result['similarity']:.3f}):\n"
            response += f"{result['chunk'][:300]}...\n\n"
        
        return response
    
    def list_files_tool(self, directory: str = ".") -> str:
        """Tool: List available files in directory"""
        try:
            files = []
            for ext in ['.pdf', '.txt', '.png', '.jpg', '.jpeg']:
                files.extend(Path(directory).glob(f"*{ext}"))
            
            if not files:
                return "No PDF, text, or image files found in the current directory."
            
            file_list = "Available files:\n"
            for file in sorted(files):
                file_list += f"- {file.name}\n"
            
            return file_list
        except Exception as e:
            return f"Error listing files: {e}"

class ReActAgent:
    """ReAct-style agent for RAG chatbot"""
    
    def __init__(self, tools: RAGTools):
        self.tools = tools
        self.conversation_history = []
    
    def think(self, observation: str) -> str:
        """Agent's thinking process"""
        return f"Thought: {observation}"
    
    def act(self, action: str, input_text: str = "") -> str:
        """Execute an action using available tools"""
        action = action.lower().strip()
        
        if action == "process_pdf":
            return self.tools.process_pdf_tool(input_text)
        elif action == "process_text":
            return self.tools.process_text_tool(input_text)
        elif action == "process_image":
            return self.tools.process_image_tool(input_text)
        elif action == "search_documents":
            return self.tools.search_documents_tool(input_text)
        elif action == "list_files":
            return self.tools.list_files_tool(input_text or ".")
        else:
            return f"Unknown action: {action}. Available actions: process_pdf, process_text, process_image, search_documents, list_files"
    
    def parse_user_intent(self, user_input: str) -> Dict[str, str]:
        """Parse user input to determine intent and extract relevant information"""
        user_input = user_input.lower().strip()
        
        # Check for file processing requests
        if any(word in user_input for word in ["process", "load", "read", "add"]):
            if ".pdf" in user_input:
                filename = re.search(r'[\w\-_]+\.pdf', user_input)
                return {"action": "process_pdf", "input": filename.group() if filename else ""}
            elif ".txt" in user_input:
                filename = re.search(r'[\w\-_]+\.txt', user_input)
                return {"action": "process_text", "input": filename.group() if filename else ""}
            elif any(ext in user_input for ext in [".png", ".jpg", ".jpeg"]):
                filename = re.search(r'[\w\-_]+\.(png|jpg|jpeg)', user_input)
                return {"action": "process_image", "input": filename.group() if filename else ""}
        
        # Check for file listing requests
        if any(word in user_input for word in ["list", "show", "files", "available"]):
            return {"action": "list_files", "input": ""}
        
        # Default to document search
        return {"action": "search_documents", "input": user_input}
    
    def respond(self, user_input: str) -> str:
        """Generate response using ReAct approach"""
        try:
            # Parse user intent
            intent = self.parse_user_intent(user_input)
            
            # Think about the action
            thought = self.think(f"User wants to {intent['action']} with input: {intent['input']}")
            
            # Execute action
            observation = self.act(intent['action'], intent['input'])
            
            # Add to conversation history
            self.conversation_history.append({
                "user": user_input,
                "thought": thought,
                "action": intent['action'],
                "observation": observation
            })
            
            return observation
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_chatbot():
    """Create and configure the chatbot components"""
    processor = DocumentProcessor()
    store = DocumentStore()
    tools = RAGTools(processor, store)
    agent = ReActAgent(tools)
    
    def chat_response(message, history):
        """Handle chat responses"""
        if not message.strip():
            return "", history
        
        response = agent.respond(message)
        history.append([message, response])
        return "", history
    
    def clear_history():
        """Clear conversation history"""
        agent.conversation_history = []
        return []
    
    # Create Gradio interface
    with gr.Blocks(title="RAG Chatbot Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG Chatbot Agent")
        gr.Markdown("This chatbot can process PDF, text, and image files from your directory and answer questions about their content.")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything about your documents, or tell me to process files...",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Usage Tips")
                gr.Markdown("""
                **Commands you can try:**
                - "List files" - Show available files
                - "Process document.pdf" - Load a PDF
                - "Process notes.txt" - Load a text file  
                - "Process image.png" - Extract text from image
                - "What is X?" - Search loaded documents
                - "Tell me about Y" - Ask questions about content
                
                **Supported file types:**
                - PDF files (.pdf)
                - Text files (.txt)
                - Images (.png, .jpg, .jpeg)
                """)
        
        # Event handlers
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        submit_btn.click(chat_response, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_history, outputs=[chatbot])
    
    return demo

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import PyPDF2
        import pytesseract
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install PyPDF2 pytesseract sentence-transformers pillow scikit-learn numpy gradio")
        exit(1)
    
    # Create and launch the chatbot
    demo = create_chatbot()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )