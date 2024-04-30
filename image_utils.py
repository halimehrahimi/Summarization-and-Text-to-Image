import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class StableDiff:
    """
    Class using Stable Diffusion v2 as Text to Image model.
    """
    def __init__(self, cuda=True) -> None:
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2",
                                                    torch_dtype=torch.float16)
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.pipeline = pipeline.to(self.device)

    def transform(self, input_text: str) -> Image:
        """
        Function to transform text to images.
        """
        input_texts = ["Book Cover: " + input_text] * 4 # Create four trials
        images = self.pipeline(input_texts).images
        return image_grid(images)


def image_grid(imgs: list, rows: int = 2, cols: int = 2) -> Image:
    """
    Defining function for the creation of a grid of images.
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size = (cols*w,
                                   rows*w))

    for i, img in enumerate(imgs):
        grid.paste(img, box = (i%cols*w, i // cols*h))
    return grid
