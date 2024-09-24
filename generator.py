import os
import torch
import random
import shutil
import words
from compel import Compel, ReturnedEmbeddingsType
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)
from diffusers import StableDiffusion3Pipeline
from diffusers import DPMSolverMultistepScheduler, KolorsPipeline
from diffusers import LuminaText2ImgPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


def create_dirs(dir_name="./assets"):
    """
    Create target directories if it doesn't exist
    :param: none
    :return: none
    """
    career_list = ["arts", "science", "business", "education", "sports"]
    os.makedirs(dir_name, exist_ok=True)
    for career in career_list:
        dir_path = f"{dir_name}/{career}"
        os.makedirs(dir_path, exist_ok=True)


class SimpleGenerator:
    """Simple Generator"""

    def __init__(self):
        self.pipe = None
        self.model_id = None

    def init(self):
        """
        Initialize StableDiffusion v2 pipeline
        :param: none
        :return: none
        """
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")

    def generate_template(self, prompt, output_path, sex, count):
        """
        Generate template images using Stablefusion v2
        :param prompt: prompt to generate image
        :param output_path: output directory path
        :param sex: gender option {M, F}
        :param count: number of image templates to be generated
        :return: none
        """
        negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"

        # start image generation
        print("Template Generation started")
        for i in range(0, count):
            image = self.pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
            output_file = os.path.join(output_path, f"{i}{sex}.jpg")
            image.save(output_file)

    def run(self):
        """
        Run image templates generator
        :param: none
        :return: none
        """
        self.model_id = "stabilityai/stable-diffusion-2-1"
        # self.model_id = 'SG161222/RealVisXL_V4.0' # not working
        # self.model_id = 'runwayml/stable-diffusion-v1-5'

        # initialize pipeline
        self.init()
        create_dirs()

        # compare to Leonardo.ai
        prompt = "a studio photo of a young construcation worker sitting at a modern cafe table doing some admin on his laptop."
        self.generate_template(prompt=prompt, output_path="assets", sex="M", count=1)


class ExpertGenerator:
    """Expert Generator that uses both base and refiner SDXL"""

    def __init__(self):
        self.base = None
        self.refiner = None
        self.compel_base = None
        self.compel_refiner = None
        self.n_steps = 100
        self.high_noise_frac = 0.75

    def init(self, improve=False):
        """
        Use whole base + refiner pipeline as an ensemble of experts
        :param improve: flag to use FreeU or not
        :return: none
        """
        # load base
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base.to("cuda")

        # load refiner
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

        # when using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
        # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

        # improve quality
        if improve:
            self.base.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
            # self.refiner.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

        # compel for base pipeline
        self.compel_base = Compel(
            tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
            text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        # compel for refiner pipeline
        self.compel_refiner = Compel(
            tokenizer=[self.refiner.tokenizer_2],
            text_encoder=[self.refiner.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[True],
        )

    def adjust_steps(self, steps, ratio):
        """
        Define how many steps and what % of steps to be run on each experts
        :param steps: number of denoising steps. more denoising steps typically produce higher quality images, but it’ll take longer to generate
        :param ratio: determine the percentage of denoising the high-noise timesteps in base model and denoising the low-noise timesteps in refiner model. it should be between (0 - 1)
        :return: none
        """
        # define how many steps and what % of steps to be run on each experts (80/20) here
        self.n_steps = steps
        self.high_noise_frac = ratio

    def generate_template(
        self, prompt, negative_prompt, output_file, guidance_scale=7.5
    ):
        """
        Generate template images using SDXL-1.0
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :return: none
        """
        print(f"Generating {output_file}")
        conditioning, pooled = self.compel_base(prompt)

        # run both experts
        image = self.base(
            # prompt=prompt,
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
            guidance_scale=guidance_scale,
        ).images

        conditioning, pooled = self.compel_refiner(prompt)
        image = self.refiner(
            # prompt=prompt,
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]

        # save image
        image.save(output_file)

    def run(self, prompt, output_path, sex, count=100, skip=True):
        """
        Initialize image templates generator
        :param prompt: prompt to generate images
        :param output_path: output directory path
        :param set: gender {M, F}
        :param count: number of images per career
        :param skip: flag to skip images with same name
        :return: none
        """
        # generate images
        for i in range(0, count):
            # set target file path
            output_file = os.path.join(output_path, f"{i}{sex}.jpg")
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    output_file=output_file,
                )


class SD3Generator:
    """Stable Diffusion 3 Generator"""

    def __init__(self):
        self.pipe = None
        self.n_steps = 25

    def init(self):
        """
        Initialize StableDiffusion v3 pipeline
        :param: none
        :return: none
        """
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
            device_map="auto",
        )

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=text_encoder,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
        # self.pipe.to("cuda")
        # self.pipe.reset_device_map()
        # self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=False)

    def adjust_steps(self, steps):
        """
        Define how many steps to be run on each generation
        :param steps: number of denoising steps. more denoising steps typically produce higher quality images, but it'll take longer to generate
        :return: none
        """
        # define how many steps and what % of steps to be run on each experts (80/20) here
        self.n_steps = steps

    def generate_template(
        self, prompt, negative_prompt, output_file, guidance_scale=4.5, short_prompt=""
    ):
        """
        Generate template images using SD3
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :param short_prompt: short prompt to be sent to CLIP encoder
        :return: none
        """
        # run inference
        print(f"[MAIN] Generating {output_file}")
        image = self.pipe(
            prompt=short_prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            height=1024,
            width=1024,
        ).images[0]

        # save image
        image.save(output_file)

    def run(self, prompt, output_path, sex, count=100, skip=True):
        """
        Initialize image templates generator
        :param prompt: prompt to generate images
        :param output_path: output directory path
        :param set: gender {M, F}
        :param count: number of images per career
        :param skip: flag to skip images with same name
        :return: none
        """
        # generate images
        for i in range(0, count):
            # set target file path
            output_file = os.path.join(output_path, f"{i}{sex}.jpg")
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    output_file=output_file,
                )


class LuminaGenerator:
    """Lumina-T2X Generator"""

    def __init__(self):
        self.pipe = None
        self.n_steps = 25

    def init(self, enable_torch_compile=False):
        """
        Initialize Lumina pipeline
        :param enable_torch_compile: flag to enable torch compilation
        :return: none
        """
        # load the pipeline
        model_id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"
        self.pipe = LuminaText2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to("cuda")

        if enable_torch_compile:
            # change the memory layout of the pipelines
            self.pipe.transformer.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)

            # compile the components
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="max-autotune", fullgraph=True
            )
            self.pipe.vae.decode = torch.compile(
                self.pipe.vae.decode, mode="max-autotune", fullgraph=True
            )

        # enable memory optimizations
        self.pipe.enable_model_cpu_offload()

    def adjust_steps(self, steps):
        """
        Define how many steps to be run on each generation
        :param steps: number of denoising steps. more denoising steps typically produce higher quality images, but it'll take longer to generate
        :return: none
        """
        # define inference steps
        self.n_steps = steps

    def generate_template(
        self, prompt, negative_prompt, output_file, guidance_scale=4.5
    ):
        """
        Generate template images using Lumina-T2X
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :return: none
        """
        # run inference
        print(f"[MAIN] Generating {output_file}")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=self.n_steps,
            height=1024,
            width=1024,
        ).images[0]

        # save image
        image.save(output_file)

    def run(self, prompt, output_path, sex, count=100, skip=True):
        """
        Initialize image templates generator
        :param prompt: prompt to generate images
        :param output_path: output directory path
        :param set: gender {M, F}
        :param count: number of images per career
        :param skip: flag to skip images with same name
        :return: none
        """
        # generate images
        for i in range(0, count):
            # set target file path
            output_file = os.path.join(output_path, f"{i}{sex}.jpg")
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    output_file=output_file,
                )


class KolorsGenerator:
    """Kolors Generator"""

    def __init__(self):
        self.pipe = None
        self.n_steps = 25

    def init(self):
        """
        Initialize Kolors pipeline
        :param: none
        :return: none
        """
        model_id = "Kwai-Kolors/Kolors-diffusers"
        self.pipe = KolorsPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )
        self.pipe.enable_model_cpu_offload()

    def adjust_steps(self, steps):
        """
        Define how many steps to be run on each generation
        :param steps: number of denoising steps. more denoising steps typically produce higher quality images, but it'll take longer to generate
        :return: none
        """
        # define inference steps
        self.n_steps = steps

    def generate_template(
        self, prompt, negative_prompt, output_file, guidance_scale=4.5
    ):
        """
        Generate template images using Kolors
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :return: none
        """
        # run inference
        print(f"[MAIN] Generating {output_file}")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=self.n_steps,
            height=1024,
            width=1024,
        ).images[0]

        # save image
        image.save(output_file)

    def run(self, prompt, output_path, sex, count=100, skip=True):
        """
        Initialize image templates generator
        :param prompt: prompt to generate images
        :param output_path: output directory path
        :param set: gender {M, F}
        :param count: number of images per career
        :param skip: flag to skip images with same name
        :return: none
        """
        # generate images
        for i in range(0, count):
            # set target file path
            output_file = os.path.join(output_path, f"{i}{sex}.jpg")
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    output_file=output_file,
                )


class ImageRefiner:
    """Image Refiner that uses refiner SDXL (Image to Image)"""

    def __init__(self):
        self.refiner = None
        self.compel = None
        self.n_steps = 100
        self.strength = 0.75

    def init(self):
        """
        Use refiner pipeline
        :param: none
        :return: none
        """
        # load refiner
        model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.refiner = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.refiner.enable_model_cpu_offload()

        # load compel
        # self.compel = Compel(
        #     tokenizer=self.refiner.tokenizer,
        #     text_encoder=self.refiner.text_encoder
        # )

    def adjust_steps(self, steps, ratio):
        """
        Strength is one of the most important parameters to consider and it'll have a huge impact on your generated image.
        It determines how much the generated image resembles the initial image. In other words:
        📈 a higher strength value gives the model more "creativity" to generate an image that's different from the initial image.
        📉 a lower strength value means the generated image is more similar to the initial image
        A strength value of 1.0 means the initial image is more or less ignored
        The strength and num_inference_steps parameters are related because strength determines the number of noise steps to add.
        For example, if the num_inference_steps is 50 and strength is 0.8, then this means adding 40 (50 * 0.8) steps of noise to the initial image
        and then denoising for 40 steps to get the newly generated image.
        :param steps: number of inference steps (denoising)
        :param ratio: strength value
        :return: none
        """
        self.n_steps = steps
        self.strength = ratio

    def generate_template(
        self, prompt, negative_prompt, output_file, init_image, guidance_scale=8.0
    ):
        """
        Generate template images using SDXL-1.0
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param init_image: input image to be refined
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :return: none
        """
        print(f"Generating {output_file}")

        # Combine guidance_scale with strength for even more precise control over how expressive the model is.
        # For example, combine a high strength + guidance_scale for maximum creativity.
        # Or use a combination of low strength and low guidance_scale to generate an image that resembles the initial image but is not as strictly bound to the prompt.
        image = self.refiner(
            # prompt_embeds = self.compel(prompt),
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            guidance_scale=guidance_scale,
            num_inference_steps=self.n_steps,
            strength=self.strength,
        ).images[0]

        # save image
        image.save(output_file)


def rename(dir_path, sex="M", start=0):
    """
    Rename files inside the folder
    :param dir_path: directory path
    :param sex: gender option {'M', 'F'}
    :param start: start index
    :return: none
    """
    # set target directory path
    target_dir = os.path.join(dir_path, "aligned")
    os.makedirs(target_dir, exist_ok=True)

    # set file index
    file_idx = start
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            target_path = os.path.join(target_dir, f"{file_idx}{sex}.jpg")
            shutil.move(file_path, target_path)
            file_idx = file_idx + 1

    print("[MAIN] Rename finished!")


def demo(count, model="sd3"):
    """
    Demo version
    :param count: number of images to be generated
    :param model: inference model option
    :return: none
    """
    demo_prompts = [
        "Behind the girl's back is a snow-covered mountain. She is a photorealistic, extraordinarily beautiful young girl with a unique appearance. Dressed in a red Chinese long dress, she has very long white hair and wears silver jewelry on her head with fantasy makeup. Her face is well highlighted, giving her a god-like, magical presence. The full-length portrait of the girl is a masterpiece. Through the fog, you can see a huge gate adorned with flowers, with more fog beyond it. She holds a Sakura flower in her hand. The girl is inspired by Dilraba, featuring a light-skinned girl with white hair and brown eyes. She is incredibly beautiful with very expressive eyes. Imagine a fictional character dressed in stunning Chinese attire, set in a winter landscape with falling snow. The ultra-realistic photography captures every detail, from her highly detailed skin texture to the overall scene's serene beauty.",
        "A realistic photo of a joyfully beaming teenage girl with a stylish pixie cut and a cozy flannel shirt, perched on a moss-covered tree stump. Her eyes sparkle with delight, capturing her undeniable connection to nature and radiating pure happiness. She’s slightly chubby with a round, natural-looking face, slight freckles, and cheeks and nose reddened by the fresh air. The vividly detailed photograph showcases the vibrant colors of the forest backdrop, enhancing the girl's lively expression. Every detail, from her sun-kissed hair to the serene surroundings, evokes a sense of tranquility and youthful exuberance.",
        "Indonesian celebrity style, bathed in cinematic lighting. Epic masterpiece, 64K resolution, with a flawless face and stunning eyes. In dramatic lighting, a stunning celebrity figure. Pay attention to the facial details and body—it must be extremely beautiful and flawless. Concept: Create a cinematic shot of Chelsea Islan. Details: Indonesian nature. Concept: Random. Face: Sun-kissed, content. Chelsea has an oval-shaped face with expressive dark brown eyes, a refined nose, and a warm, inviting smile. Her medium-length, dark brown hair is often styled in soft waves or sleek, straight looks. Clothing: Chelsea’s fashion sense is elegant and polished. She might wear a high-waisted, knee-length pencil skirt in a solid color, paired with a fitted white blouse with subtle lace detailing. She accessorizes with pearl earrings, a delicate bracelet, and classic black pumps. Composition: Peaceful, rural, serene.",
        "robert pattinson, handsome, smiling, dressed as an astronaut from prometheus, Existenz Neo-noir film directed by David Cronenberg and Ridley Scott",
        "A photo of the guy with hook hands holding a fish. He has long, dark hair pulled back in a bun and looks straight ahead with his head slightly tilted to the left. The shot features soft natural lighting on a sunny day with natural colors, backlit, and taken on Kodak Gold 400 film.",
        "A studio photo of a man sitting on a blue sofa, with a modern Australian home office behind him. On the desk is an open MacBook, notebooks, and other general accessories. The office's color scheme includes warm grey, lilac, yellow, and white.",
        "A studio photo of a young construction worker sitting at a modern cafe table, doing some admin work on his laptop.",
        "An upper body photo of a beautiful Chinese girl walking on the road. It's summer, and many trees line the street. There are many potted plants placed along the way. She has beautiful eyes and is smiling. She is wearing a golden bracelet and a nice necklace, along with blue jeans trousers and a light brown jacket with zipper. She also wears a white T-shirt under the jacket.",
        "A Chinese beauty, She has regular features, a good face, decent makeup, a curvy figure, black hair, a white gauze bandage vest and shorts, orthopedic protective gear for both arms, leg straps and exoskeleton prosthetics, and a white exoskeleton flying mecha. her body. She stood by the pool on the cruise ship, smiling and getting ready to go. Full body images, large high resolution images, bright colors",
    ]

    create_dirs(dir_name="./assets_sd3")

    # init inference runner
    generator = None
    if model == "sdxl":
        # use sdxl - acceptable quality
        generator = ExpertGenerator()
        generator.init(improve=False)
        generator.adjust_steps(steps=100, ratio=0.7)
    elif model == "sd3":
        # use sd3-medium model - good quality
        generator = SD3Generator()
        generator.init()
        generator.adjust_steps(steps=28)
    elif model == "kolors":
        # use kolors model - unrealistic
        generator = KolorsGenerator()
        generator.init()
        generator.adjust_steps(steps=50)
    elif model == "lumina":
        # use lumina model - need to test more
        generator = LuminaGenerator()
        generator.init()
        generator.adjust_steps(steps=30)
    else:
        # not verified yet
        generator = SimpleGenerator()
        generator.init()

    # run generator
    if generator is not None:
        negative_prompt = "eyeglasses, extra fingers, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"

        # set guidance scale
        """
        A value of 0 gives random images and does not consider your prompt at all.
        Lower values give more "creative" images, but with a lot of elements unrelated to your prompt (pulled from other unrelated images). Prompt itself may also be poorly represented.
        Higher values give images which represent your prompt more precisely. However, results are less "creative" (less elements are pulled from images unrelated to your prompt), and with particularly high values you cat get some image artifacts.
        Negative values, in theory, can also work - making images of everything but things in your prompt. I don't think it currently works with SD though.

        With SDXL, optimal values are between 6-12. Use lower values for creative outputs, and higher values if you want to get more usable, sharp images.
        """
        prompts = words.description["arts"]["F"]["hairstyle"]
        value = 3.6  # for SD3
        # value = 6.5 # for Kolors
        # value = 4.0 # for Lumina-T2X
        start = 0  # this needs to be changed accordingly
        for i in range(start, start + count):
            id = random.randint(0, len(prompts) - 1)  # generate random number
            # id = 0
            # output_file = os.path.join('./output', f'{i}M.jpg')
            output_file = os.path.join("./output", f"{i}F.jpg")
            generator.generate_template(
                prompt=prompts[id],
                negative_prompt=negative_prompt,
                output_file=output_file,
                guidance_scale=value,
            )


def refine(count):
    """
    Refine image
    :param count: number of images to be generated
    :return: none
    """
    # use SDXL image-to-image refiner
    refiner = ImageRefiner()
    refiner.init()
    refiner.adjust_steps(steps=30, ratio=0.25)

    # set prompts
    negative_prompt = "eyeglasses, extra fingers, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
    prompts = words.description["arts"]["F"]["hands_behind"]

    # get available file list
    dir_path = "D:/AI/Dataset/Celebrity Recognition DB_65/Bai Lu_004206_F"
    img_files = os.listdir(dir_path)
    images = [file for file in img_files if file.endswith(("jpg", "png"))]

    # initialize
    value = 4.0  # guidance scale
    start = 0  # this needs to be changed accordingly
    for i in range(start, start + count):
        # get random image inside the folder
        num = random.randint(0, len(images) - 1)
        img_name = "Bai Lu_XdAE6x_5c_20240617214812 (100).jpg"
        # img_path = os.path.join(dir_path, images[num])
        img_path = os.path.join(dir_path, img_name)
        img = load_image(img_path)

        # set output file path
        id = random.randint(0, len(prompts) - 1)  # generate random number
        # output_file = os.path.join('./output', f'{i}M.jpg')
        output_file = os.path.join("./output", f"{i}F.jpg")

        # generate image
        refiner.generate_template(
            prompt=prompts[id],
            negative_prompt=negative_prompt,
            output_file=output_file,
            init_image=img,
            guidance_scale=value,
        )


if __name__ == "__main__":
    # rename("./output", "M", start=0)
    # rename("./output/cropped/male_educator", "M", start=190)
    # demo(40, model="kolors")
    # demo(30, model="lumina")
    # demo(600, model="sd3")
    # demo(150, model="sdxl")
    refine(5)
