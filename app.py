import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import pdfplumber
from PIL import Image
import io
import json
from fastapi import Request

# Initialize FastAPI and Gradio
app = FastAPI()

# Load the models
text_model_name = "microsoft/phi-2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForCausalLM.from_pretrained(text_model_name, torch_dtype=torch.float16, device_map="auto")
text_pipe = pipeline("text-generation", model=text_model, tokenizer=text_tokenizer)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

diagram_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# Function to handle text input
def user_text_input(text):
    result = text_pipe(text, max_length=300, do_sample=True)
    return result[0]['generated_text']

# Function to handle image input
def user_image_input(image):
    inputs = blip_processor(image, return_tensors="pt").to("cuda")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return f"Image Analysis: {caption}"

# Function to handle document input
def user_document_input(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return user_text_input(text[:1000])

# Function to generate diagrams
def generate_diagram(prompt):
    image = diagram_pipe(prompt).images[0]
    output_path = "/mnt/data/outputs/generated_diagram.png"
    image.save(output_path)
    return output_path

# Main controller function for Gradio app and API
def study_buddy_controller(text, image=None, document=None, gen_diagram_prompt=None):
    results = []

    if text and not image and not document:
        results.append(user_text_input(text))

    if image:
        results.append(user_image_input(image))

    if document:
        results.append(user_document_input(document.name))

    if gen_diagram_prompt:
        diagram_path = generate_diagram(gen_diagram_prompt)
        results.append(f"Diagram generated at: {diagram_path}")

    return "\n\n---\n\n".join(results)

# Define the Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### 🤖 Study Buddy AI (Text, Image, Diagram, PDF)")

        with gr.Row():
            text_input = gr.Textbox(label="Enter your question / explanation request")
            image_input = gr.Image(type="pil", label="Upload Image (e.g., diagram, circuit)")
            doc_input = gr.File(label="Upload PDF Notes", file_types=[".pdf"])
            diagram_prompt = gr.Textbox(label="Prompt to generate a diagram (e.g., 'Bohr’s atom model')")

        output = gr.Textbox(label="AI Output")
        run_button = gr.Button("Run Study Buddy")

        run_button.click(fn=study_buddy_controller, 
                         inputs=[text_input, image_input, doc_input, diagram_prompt], 
                         outputs=[output])

    return demo

gradio_interface = create_gradio_interface()

# Launch Gradio interface
gradio_interface.launch(share=True)

# Define API routes using FastAPI
@app.post("/ask")
async def ask(request: Request):
    """
    POST request to interact with the Study Buddy AI.
    Accepts JSON payload containing `text`, `image`, `document`, and `gen_diagram_prompt`.
    """
    body = await request.json()
    text = body.get("text", None)
    image = body.get("image", None)
    document = body.get("document", None)
    gen_diagram_prompt = body.get("gen_diagram_prompt", None)

    # Handle inputs and generate response
    try:
        response = study_buddy_controller(text, image, document, gen_diagram_prompt)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    """
    Basic root endpoint for testing the API.
    """
    return {"message": "Welcome to the Study Buddy API. Use POST /ask to interact with the app."}
