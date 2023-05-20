from pathlib import Path

import gradio as gr
import upscaler
import tempfile
import atexit
import uuid

# Temporary directory for output images
TEMP_PATH = tempfile.TemporaryDirectory()
atexit.register(TEMP_PATH.cleanup)


def get_temp_name(uid_length: int = 8) -> Path:
    return Path(TEMP_PATH.name) / f"tmp_{uuid.uuid4().hex[:uid_length]}.png"


def upscale_image(
    prompt,
    negative_prompt,
    rows=3,
    seed=0,
    image=None,
    enable_custom_sliders=False,
    guidance=7,
    iterations=50,
    xformers_input=False,
    cpu_offload_input=False,
    attention_slicing_input=False,
) -> str:
    cols = rows
    output_image = upscaler.upscale_image(
        image,
        int(rows),
        int(cols),
        int(seed),
        prompt,
        negative_prompt,
        xformers_input,
        cpu_offload_input,
        attention_slicing_input,
        enable_custom_sliders,
        guidance,
        iterations,
    )

    output_image_path = get_temp_name()
    output_image.save(output_image_path)
    return str(output_image_path)


with gr.Blocks() as block:
    image_input = gr.Image(label="Input Image")
    with gr.Row():
        image_width_input = gr.Number(value=512, interactive=True, label="Source Image Width")
        image_height_input = gr.Number(value=512, interactive=True, label="Source Image Height")

    @image_input.change
    def on_image_change():
        """Update the image width and height when the image is changed."""
        image_width_input.value = image_input.value.shape[1]
        image_height_input.value = image_input.value.shape[0]


    prompt_input = gr.Textbox(label="Prompt")
    negative_prompt_input = gr.Textbox(label="Negative prompt")
    seed_input = gr.Number(-1, label="Seed")
    row_input = gr.Number(
        1, label="Tile grid dimension amount (number of rows and columns) - v x v"
    )
    xformers_input = gr.Checkbox(
        True, label="Enable Xformers memory efficient attention"
    )
    enable_custom_sliders = gr.Checkbox(
        False,
        label="(NOT RECOMMENDED) Click to enable the sliders below; if unchecked, it will ignore them and use the default settings",
    )
    cpu_offload_input = gr.Checkbox(
        True, label="Enable sequential CPU offload"
    )
    attention_slicing_input = gr.Checkbox(
        True, label="Enable attention slicing"
    )
    output_image = gr.Image(label="Output Image", type="pil")
    guidance = gr.Slider(
        2, 15, 7, step=1, label="Guidance Scale: How much the AI influences the Upscaling."
    )
    iterations = gr.Slider(10, 75, 50, step=1, label="Number of Iterations")

    button = gr.Button("Upscale Image")
    button.click(
        fn=upscale_image,
        inputs=[
            prompt_input,
            negative_prompt_input,
            row_input,
            seed_input,
            image_input,
            enable_custom_sliders,
            guidance,
            iterations,
            xformers_input,
            cpu_offload_input,
            attention_slicing_input,
        ],
        outputs=[output_image],
    )

block.launch(server_port=7865)
