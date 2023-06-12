from typing import Optional

import gradio as gr
import qrcode
import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
)
from gradio.components import Image, Radio, Slider, Textbox, Number
from PIL import Image as PilImage
from typing_extensions import Literal


def main():
    device = (
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    )

    controlnet_tile = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=False,
        cache_dir="./cache"
    ).to(device)

    controlnet_brightness  = ControlNetModel.from_pretrained(
        "ioclab/control_v1p_sd15_brightness",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        cache_dir="./cache"
    ).to(device)

    def make_pipe(hf_repo: str, device: str) -> StableDiffusionControlNetPipeline:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            hf_repo,
            controlnet=[controlnet_tile, controlnet_brightness],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir="./cache",
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe.to(device)

    pipes = {
        "DreamShaper": make_pipe("Lykon/DreamShaper", device),
        # "DreamShaper": make_pipe("Lykon/DreamShaper", "cpu"),
        # "Realistic Vision V1.4": make_pipe("SG161222/Realistic_Vision_V1.4", "cpu"),
        # "OpenJourney": make_pipe("prompthero/openjourney", "cpu"),
        # "Anything V3": make_pipe("Linaqruf/anything-v3.0", "cpu"),
    }

    def move_pipe(hf_repo: str):
        for pipe_name, pipe in pipes.items():
            if pipe_name != hf_repo:
                pipe.to("cpu")
        return pipes[hf_repo].to(device)

    def predict(
        model: Literal[
            "DreamShaper",
            # "Realistic Vision V1.4",
            # "OpenJourney",
            # "Anything V3"
        ],
        qrcode_data: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 100,
        guidance_scale: int = 9,
        controlnet_conditioning_tile: float = 0.25,
        controlnet_conditioning_brightness: float = 0.45,
        seed: int = 1331,
    ) -> PilImage:
        generator = torch.Generator(device).manual_seed(seed)
        if model == "DreamShaper":
            pipe = pipes["DreamShaper"]
            # pipe = move_pipe("DreamShaper Vision V1.4")
        # elif model == "Realistic Vision V1.4":
        #     pipe = move_pipe("Realistic Vision V1.4")
        # elif model == "OpenJourney":
        #     pipe = move_pipe("OpenJourney")
        # elif model == "Anything V3":
        #     pipe = move_pipe("Anything V3")

        
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=11,
            border=9,
        )
        qr.add_data(qrcode_data)
        qr.make(fit=True)
        qrcode_image = qr.make_image(
            fill_color="black",
            back_color="white"
        ).convert("RGB")
        qrcode_image = qrcode_image.resize((512, 512), PilImage.LANCZOS)

        image = pipe(
            prompt,
            [qrcode_image, qrcode_image],
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[
                controlnet_conditioning_tile,
                controlnet_conditioning_brightness
            ]
        ).images[0]

        return image


    ui = gr.Interface(
        fn=predict,
        inputs=[
            Radio(
                value="DreamShaper",
                label="Model",
                choices=[
                    "DreamShaper",
                    # "Realistic Vision V1.4",
                    # "OpenJourney",
                    # "Anything V3"
                ],
            ),
            Textbox(
                value="https://twitter.com/JulienBlanchon",
                label="QR Code Data",
            ),
            Textbox(
                value="Japanese ramen with chopsticks, egg and steam, ultra detailed 8k",
                label="Prompt",
            ),
            Textbox(
                value="logo, watermark, signature, text, BadDream, UnrealisticDream",
                label="Negative Prompt",
                optional=True
            ),
            Slider(
                value=100,
                label="Number of Inference Steps",
                minimum=10,
                maximum=400,
                step=1,
            ),
            Slider(
                value=9,
                label="Guidance Scale",
                minimum=1,
                maximum=20,
                step=1,
            ),
            Slider(
                value=0.25,
                label="Controlnet Conditioning Tile",
                minimum=0.0,
                maximum=1.0,
                step=0.05,

            ),
            Slider(
                value=0.45,
                label="Controlnet Conditioning Brightness",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            Number(
                value=1,
                label="Seed",
                precision=0,
            ),

        ],
        outputs=Image(
            label="Generated Image",
            type="pil",
        ),
        examples=[
            [
                "DreamShaper",
                "https://twitter.com/JulienBlanchon",
                "rock, mountain",
                "",
                100,
                9,
                0.25,
                0.45,
                1,
            ],
            [
                "DreamShaper",
                "https://twitter.com/JulienBlanchon",
                "Japanese ramen with chopsticks, egg and steam, ultra detailed 8k",
                "logo, watermark, signature, text, BadDream, UnrealisticDream",
                100,
                9,
                0.25,
                0.45,
                1,
            ],
            # [
            #     "Anything V3",
            #     "https://twitter.com/JulienBlanchon",
            #     "Japanese ramen with chopsticks, egg and steam, ultra detailed 8k",
            #     "logo, watermark, signature, text, BadDream, UnrealisticDream",
            #     100,
            #     9,
            #     0.25,
            #     0.60,
            #     1,
            # ],
            [
                "DreamShaper",
                "https://twitter.com/JulienBlanchon",
                "processor, chipset, electricity, black and white board",
                "logo, watermark, signature, text, BadDream, UnrealisticDream",
                300,
                9,
                0.50,
                0.30,
                1,
            ],
        ],
        cache_examples=True,
        title="Stable Diffusion QR Code Controlnet",
        description="Generate QR Code with Stable Diffusion and Controlnet",
        allow_flagging="never",
        max_batch_size=1,
    )

    ui.queue(concurrency_count=10).launch()

if __name__ == "__main__":
    main()