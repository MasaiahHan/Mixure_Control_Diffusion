import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "darkstorm2150/Protogen_x5.8_Official_Release"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "a photo of the dolomites"
image = pipe(prompt)