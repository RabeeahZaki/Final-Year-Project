# authtoken.py

# How to obtain an authentication token: https://huggingface.co/docs/hub/security-tokens
auth_token = "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"

# Libraries for building GUI
import tkinter as tkpi
import customtkinter as ctk

# Machine Learning libraries
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Libraries for processing images
from PIL import ImageTk

# Private module
#from authtoken import auth_token

# Create the application's user interface
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image App")
app.configure(bg='black')
ctk.set_appearance_mode("dark")

# Create an input box on the user interface
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 15), text_color="white", fg_color="black")
prompt.place(x=10, y=10)

# Create a placeholder to display the generated image
img_placeholder = ctk.CTkLabel(height=512, width=512, text="")
img_placeholder.place(x=10, y=110)

# Download the stable diffusion model from Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
stable_diffusion_model.to(device)

# Generate an image from text
def generate_image():
    """This function generates an image from text using stable diffusion."""
    with autocast(device):
        image = stable_diffusion_model(prompt.get(), guidance_scale=8.5)["sample"][0]

    # Save the generated image
    image.save('generatedimage.png')

    # Display the generated image on the user interface
    img = ImageTk.PhotoImage(image)
    img_placeholder.configure(image=img)

# Create a button to trigger image generation
trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 15), text_color="black", fg_color="white",
                        command=generate_image)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
