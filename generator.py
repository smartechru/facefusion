import os
import torch
import random
import shutil
from compel import Compel, ReturnedEmbeddingsType
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers import StableDiffusion3Pipeline


def create_dirs(dir_name='./assets'):
    """
    Create target directories if it doesn't exist
    :param: none
    :return: none
    """
    career_list = [ 'arts', 'science', 'business', 'education', 'sports' ]
    os.makedirs(dir_name, exist_ok=True)
    for career in career_list:
        dir_path = f'{dir_name}/{career}'
        os.makedirs(dir_path, exist_ok=True)


class SimpleGenerator:
    """ Simple Generator """

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
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
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
        print('Template Generation started')
        for i in range(0, count):
            image = self.pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
            output_file = os.path.join(output_path, f'{i}{sex}.jpg')
            image.save(output_file)

    def run(self):
        """
        Run image templates generator
        :param: none
        :return: none
        """
        self.model_id = 'stabilityai/stable-diffusion-2-1'
        # self.model_id = 'SG161222/RealVisXL_V4.0' # not working
        # self.model_id = 'runwayml/stable-diffusion-v1-5'

        # initialize pipeline
        self.init()
        create_dirs()

        # compare to Leonardo.ai
        prompt = "a studio photo of a young construcation worker sitting at a modern cafe table doing some admin on his laptop."
        self.generate_template(prompt=prompt, output_path='assets', sex='M', count=1)


class ExpertGenerator:
    """ Expert Generator that uses both base and refiner SDXL """

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
            use_safetensors=True
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
            requires_pooled=[False, True]
        )

        # compel for refiner pipeline
        self.compel_refiner = Compel(
            tokenizer=[self.refiner.tokenizer_2],
            text_encoder=[self.refiner.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[True]
        )

    def adjust_steps(self, steps, ratio):
        """
        Define how many steps and what % of steps to be run on each experts
        :param steps: number of denoising steps. more denoising steps typically produce higher quality images, but it’ll take longer to generate
        :param ratio: determine the percentage of denoising the high-noise timesteps in base model and denoising the low-noise timesteps in refiner model. it should be between (0 - 1)
        :return: none
        """
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        self.n_steps = steps
        self.high_noise_frac = ratio

    def generate_template(self, prompt, negative_prompt, output_file, guidance_scale=7.5):
        """
        Generate template images using SDXL-1.0
        :param prompt: prompt to generate image
        :param negative_prompt: negative prompt to generate image
        :param output_file: output file path
        :param guidance_scale: parameter to determine how much the prompt influences image generation
        :return: none
        """
        print(f'Generating {output_file}')
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

    def run(self, prompt, output_path, sex, count = 100, skip=True):
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
            output_file = os.path.join(output_path, f'{i}{sex}.jpg')
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(prompt=prompt, negative_prompt=negative_prompt, output_file=output_file)


class SD3Generator:
    """ Stable Diffusion 3 Generator """

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
            device_map="balanced"
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
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        self.n_steps = steps

    def generate_template(self, prompt, negative_prompt, output_file, guidance_scale=4.5, short_prompt=""):
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
        print(f'[MAIN] Generating {output_file}')
        image = self.pipe(
            prompt=short_prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]

        # save image
        image.save(output_file)

    def run(self, prompt, output_path, sex, count = 100, skip=True):
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
            output_file = os.path.join(output_path, f'{i}{sex}.jpg')
            if skip and os.path.isfile(output_file):
                # this is an existing file
                pass
            else:
                negative_prompt = "eyeglasses, extra arms, wonky hands, ugly, deformed, disfigured, poor details, bad anatomy"
                self.generate_template(prompt=prompt, negative_prompt=negative_prompt, output_file=output_file)


def batch_generation(count=100):
    """
    Batch process to generate images
    :param count: batch size for each career
    :return: none
    """
    prompts = [
        "a natural upper body photo of a Chinese actor in his 20s wearing a nice jacket",
        "a natural upper body photo of a Chinese actress in her 20s wearing a beautiful skirt",
        "a natural upper body photo of a Chinese male scientist in his 20s",
        "a natural upper body photo of a Chinese female scientist in her 20s",
        "a color photo of Chinese male educator in his 20s",
        "a color photo of Chinese female educator in her 20s",
        "a color photo of Chinese sportsman in his 20s",
        "a color photo of Chinese sportswoman in her 20s",
        "a natural upper body photo of a Chinese businessman in his 20s",
        "a natural upper body photo of a Chinese businesswoman in her 20s",
    ]

    # create output directory
    create_dirs()

    # init expert
    expert = ExpertGenerator()
    expert.init()

    # run expert
    # expert.run(prompt=prompts[0], output_path='./assets/artist', sex='M', count=count)
    # expert.run(prompt=prompts[1], output_path='./assets/artist', sex='F', count=count)
    # expert.run(prompt=prompts[2], output_path='./assets/scientist', sex='M', count=count)
    # expert.run(prompt=prompts[3], output_path='./assets/scientist', sex='F', count=count)
    # expert.run(prompt=prompts[4], output_path='./assets/educator', sex='M', count=count)
    # expert.run(prompt=prompts[5], output_path='./assets/educator', sex='F', count=count)
    # expert.run(prompt=prompts[6], output_path='./assets/sports', sex='M', count=count)
    # expert.run(prompt=prompts[7], output_path='./assets/sports', sex='F', count=count)
    expert.run(prompt=prompts[8], output_path='./assets/business', sex='M', count=count)
    # expert.run(prompt=prompts[9], output_path='./assets/business', sex='F', count=count)


def rename(dir_path, sex='M', start=0):
    """
    Rename files inside the folder
    :param dir_path: directory path
    :param sex: gender option {'M', 'F'}
    :param start: start index
    :return: none
    """
    # set target directory path
    target_dir = os.path.join(dir_path, 'aligned')
    os.makedirs(target_dir, exist_ok=True)

    # set file index
    file_idx = start
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            target_path = os.path.join(target_dir, f'{file_idx}{sex}.jpg')
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

    ### actress
    prompts_actress = [
        "Spring, A Chinese actress is on the road wearing casual clothes (it could be a dress, suit, t-shirt, jacket, sweater, etc.) and smiling. She is a light-skinned girl with dark brown hair and brown eyes. She is a lovely girl with very expressive eyes. She is now surrounded by beautiful flowers. The photo must be natural and realistic and the background must change randomly.",
        "Summer, stage scene. A Chinese actress is standing in the forest wearing casual clothes (it could be a dress, suit, t-shirt, etc.) and smiling. She is a light-skinned girl with dark brown hair and brown eyes. She is a lovely girl with very expressive eyes. The photo must be natural and realistic and the background must change randomly.",
        "Autumn, camera scene. A Chinese actress is on the crowed road wearing casual clothes (it could be a dress, suit, jacket, sweater, etc.) and smiling. She is a light-skinned girl with dark brown hair and brown eyes. She is a lovely girl with very expressive eyes. The photo must be natural and realistic and the background must change randomly.",
        "Winter, it is possible to set a snow scenery such as a snow-covered mountain or a snowy road. The photo must be natural and realistic. A beautiful Chinese girl with dark brown hair and brown eyes is smiling. This light-skinned girl is very lovely with very expressive eyes. In her winter clothes, she looks very cute. She could wear a beanie.",
        "An indoor stage scene, A beautiful Chinese pianist is playing the piano in casual clothing (this can be a dress, jacket, sweater, shirt, and so on). She is a light-skinned girl with dark brown hair and brown eyes. She is a pretty girl with very expressive eyes. The photo should be natural and realistic.",
        "A beautiful Chinese actress is holding a beautiful rose in her hand. The color of the rose could be yellow, white, pink, or red. She is wearing casual clothes (it could be a dress, jacket, sweater, t-shirt, winter clothes, etc.) and smiling. She is a light-skinned girl with dark brown hair and brown eyes. She is a lovely girl with very expressive eyes. The photo must be natural and realistic and the background must change randomly.",
    ]

    prompts_actress_hands_into_pockets = [
        # natural scene
        "In the gentle light of spring, a young Chinese woman in her 20s or 30s stands by a large window, her hands tucked into her pockets to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The soft spring light filters through the curtains, casting a warm glow on her face and highlighting her serene and thoughtful expression. She wears a light, floral dress that complements the fresh, vibrant colors of the season. A vase of freshly picked flowers sits on the table beside her, filling the room with their delicate fragrance. This photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman, harmonizing perfectly with the tranquil and rejuvenating atmosphere of spring.",
        "As summer arrives, the young Chinese woman in her 20s or 30s is found in a sun-drenched living room, her hands casually tucked into her pockets or resting on her hips, her upper body radiating warmth and vitality as she smiles brightly at the camera. She sits comfortably on a sofa, wearing a light, breezy outfit that reflects the season's heat. The large windows are open, allowing a gentle summer breeze to flow through, rustling the curtains and cooling the room. A fan hums softly in the background, adding to the relaxed, lazy afternoon ambiance. The vibrant colors of summer, with potted plants and bright cushions, frame her silhouette beautifully. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the spirit of summer.",
        # "Autumn brings a cozy, warm atmosphere as the young Chinese woman in her 20s or 30s stands by the fireplace in a richly decorated living room, her hands tucked into her pockets or gently resting on the mantle to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The flickering firelight dances across her face, enhancing her contemplative expression as she sips on a hot cup of tea. The room is adorned with autumnal decorations—pumpkins, dried leaves, and warm-toned fabrics—reflecting the season's rich hues. The soft crackling of the fire and the gentle glow it casts create a serene and inviting ambiance. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the cozy and introspective atmosphere of autumn.",
        "In the stillness of winter, the young Chinese woman in her 20s or 30s stands by a frosted window, her hands tucked into her pockets to keep her fingers hidden, her upper body reflected in the glass and smiling warmly at the camera, revealing a serene and elegant demeanor. She is clad in a warm, stylish sweater, exuding quiet strength and dignified beauty. The room is softly lit by the glow of a nearby fireplace, casting a gentle light that enhances the tranquility of the scene. Snow falls silently outside, adding to the peaceful ambiance. She holds a mug of hot cocoa, the steam rising and mingling with the warm, cozy atmosphere. This photograph, focused on her upper body, captures the serene beauty and quiet strength of young Chinese women in the tranquil winter setting.",
        "In the gentle light of spring, a young Chinese woman in her 20s or 30s stands amidst a blooming garden, her hands tucked into her pockets to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The flowers around her reflect the delicate blush of her cheeks, and her eyes, bright and curious, resemble the azure sky, clear and boundless. Each breeze that passes through the garden seems to whisper secrets to her, lifting strands of her hair in a dance with the petals that fall gently at her feet. The soft fabric of her dress flows like the gentle streams that meander through the fields, echoing the tranquility of nature itself. This serene photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman immersed in the natural beauty of spring, perfectly blending with the blooming garden.",
        "As summer arrives, the young Chinese woman in her 20s or 30s stands by a serene lake, her hands casually tucked into her pockets or resting on her hips, her upper body radiating warmth and vitality as she smiles brightly at the camera. Her light, breezy outfit reflects the season's heat, standing out against the vibrant greenery. Her smile is as captivating as the sunflowers that turn their heads to follow the light. The vibrant greens of the trees frame her silhouette beautifully, highlighting the harmony between her and the natural world. Her laughter, like the song of birds, fills the air with joy and vitality. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the spirit of summer.",
        # city scene
        "Autumn brings a tapestry of colors, and the young Chinese woman in her 20s or 30s walks through a forest adorned in hues of amber, gold, and crimson, her hands tucked into her pockets or gently resting on the leaves to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The leaves crunch softly beneath her feet, a symphony of nature that accompanies her every step. Her eyes, now deeper and more contemplative, match the richness of the season. The cool air brushes against her skin, enhancing the rosy glow of her cheeks. Her presence in the autumnal landscape is a testament to the fleeting yet profound beauty of nature's transitions. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the rich tapestry of autumn.",
        "In the stillness of winter, the young Chinese woman in her 20s or 30s stands by a frosted window, her hands tucked into her pockets to keep her fingers hidden, her upper body reflected in the glass and smiling warmly at the camera, revealing a serene and elegant demeanor. She is clad in a warm, stylish sweater, exuding quiet strength and dignified beauty. The room is softly lit by the glow of a nearby fireplace, casting a gentle light that enhances the tranquility of the scene. Snow falls silently outside, adding to the peaceful ambiance. She holds a mug of hot cocoa, the steam rising and mingling with the warm, cozy atmosphere. This photograph, focused on her upper body, captures the serene beauty and quiet strength of young Chinese women in the tranquil winter setting.",
        "In the heart of a bustling city, a young Chinese woman in her 20s or 30s stands on a quiet street corner, her hands tucked into her pockets to keep her fingers hidden, her upper body gracefully poised as she smiles warmly at the camera. The tall skyscrapers and modern architecture rise behind her, their glass facades reflecting the soft morning light. She wears a stylish, urban outfit that complements the contemporary surroundings. The street is unusually empty, allowing her presence to dominate the scene. This photograph, focused on her upper body, captures the timeless beauty and sophisticated elegance of a young Chinese woman amidst the serene early morning of the city.",
        "As the sun sets over the city, the young Chinese woman in her 20s or 30s is found on a rooftop terrace, her hands casually tucked into her pockets or resting on the railing, her upper body radiating warmth and vitality as she smiles brightly at the camera. The cityscape stretches out behind her, with the evening lights beginning to twinkle like stars. Her outfit reflects a modern and chic style, perfectly suited for a night out in the city. The rooftop garden, with its lush greenery and twinkling fairy lights, frames her silhouette beautifully. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the vibrant energy of the city at dusk.",
        "Under the neon lights of the urban night, the young Chinese woman in her 20s or 30s stands confidently on a quiet street, her hands tucked into her pockets or gently resting on her hips to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The colorful signs and lights illuminate her face, enhancing her features with a soft glow. She wears a fashionable, night-time outfit that highlights her elegance and style. The usually busy street is now calm, allowing her presence to shine against the backdrop of the vibrant city lights. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the dynamic and colorful atmosphere of the city at night.",
        # travel/resort scene
        "In a trendy urban café, a young Chinese woman in her 20s or 30s sits by the window, her hands tucked into her pockets or resting on the table to keep her fingers hidden, her upper body elegantly poised as she smiles warmly at the camera. The café is modern and stylish, with large windows letting in plenty of natural light. Her casual yet chic outfit blends seamlessly with the sophisticated décor of the café. The background shows a quiet street scene, devoid of other people, highlighting the tranquility of the moment. This photograph, focused on her upper body, captures the timeless beauty and grace of a young Chinese woman, harmonizing perfectly with the serene and contemporary atmosphere of the urban café.",
        "On a pristine beach, a young Chinese woman in her 20s or 30s stands with the ocean behind her, her hands tucked into her pockets to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The soft sand beneath her feet and the gentle waves crashing in the background create a serene and picturesque setting. She wears a light, flowy dress that complements the beach's natural colors. The sun sets behind her, casting a warm, golden glow that highlights her features. This photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman, harmonizing perfectly with the tranquil and idyllic atmosphere of a beach at sunset.",
        "In the lush greenery of a mountain retreat, the young Chinese woman in her 20s or 30s stands by a crystal-clear lake, her hands casually tucked into her pockets or resting on her hips, her upper body radiating warmth and vitality as she smiles brightly at the camera. The mountains and trees reflected in the still water behind her add a sense of peace and natural beauty to the scene. She wears comfortable, stylish outdoor attire that blends seamlessly with the surroundings. The air is fresh and crisp, enhancing the sense of tranquility. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the serene beauty of a mountain retreat.",
        # "At a vibrant, tropical resort, the young Chinese woman in her 20s or 30s stands by the pool, her hands tucked into her pockets or gently resting on the poolside railing to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The bright blue water of the pool and the lush palm trees surrounding it create a lively and luxurious backdrop. She wears a chic swimsuit cover-up that highlights her elegance and style. The clear sky and the sparkling water reflect the joyful and relaxing ambiance of the resort. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the dynamic and tropical atmosphere of the resort.",
        # "In the peaceful setting of a countryside vineyard, a young Chinese woman in her 20s or 30s stands among the rows of grapevines, her hands tucked into her pockets to keep her fingers hidden, her upper body gracefully poised as she smiles warmly at the camera. The ripe grapes and the rolling hills in the background create a picturesque and serene scene. She wears a light, summery dress that complements the natural colors of the vineyard. The soft light of the late afternoon sun bathes her in a gentle glow, enhancing her features. This photograph, focused on her upper body, captures the timeless beauty and graceful presence of a young Chinese woman, harmonizing perfectly with the peaceful and idyllic atmosphere of the vineyard.",
    ]

    prompts_actress_hands_behind_back = [
        # indoor scene
        "In the gentle light of spring, a young Chinese woman in her 20s or 30s stands by a large window, her hands gracefully positioned behind her back to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The soft spring light filters through the curtains, casting a warm glow on her face and highlighting her serene and thoughtful expression. She wears a light, floral dress that complements the fresh, vibrant colors of the season. A vase of freshly picked flowers sits on the table beside her, filling the room with their delicate fragrance. This photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman, harmonizing perfectly with the tranquil and rejuvenating atmosphere of spring.",
        "As summer arrives, the young Chinese woman in her 20s or 30s is found in a sun-drenched living room, her hands resting gently on the armrest of a chair to keep her fingers hidden, her upper body radiating warmth and vitality as she smiles brightly at the camera. She sits comfortably on a sofa, wearing a light, breezy outfit that reflects the season's heat. The large windows are open, allowing a gentle summer breeze to flow through, rustling the curtains and cooling the room. A fan hums softly in the background, adding to the relaxed, lazy afternoon ambiance. The vibrant colors of summer, with potted plants and bright cushions, frame her silhouette beautifully. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the spirit of summer.",
        "Autumn brings a cozy, warm atmosphere as the young Chinese woman in her 20s or 30s stands in a richly decorated living room, her hands gently positioned behind her back to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The soft light of a table lamp casts a warm glow across her face, enhancing her contemplative expression. The room is adorned with autumnal decorations—dried leaves, warm-toned fabrics, and a bowl of seasonal fruits—reflecting the season's rich hues. A soft blanket is draped over the arm of a nearby chair, adding to the inviting ambiance. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the cozy and introspective atmosphere of autumn.",
        "In the stillness of winter, the young Chinese woman in her 20s or 30s stands by a frosted window, her hands gently clasped together at her front to keep her fingers hidden, her upper body reflected in the glass and smiling warmly at the camera, revealing a serene and elegant demeanor. She is clad in a warm, stylish sweater, exuding quiet strength and dignified beauty. The room is softly lit by the glow of a nearby fireplace, casting a gentle light that enhances the tranquility of the scene. Snow falls silently outside, adding to the peaceful ambiance. She holds a mug of hot cocoa, the steam rising and mingling with the warm, cozy atmosphere. This photograph, focused on her upper body, captures the serene beauty and quiet strength of young Chinese women in the tranquil winter setting.",
        # natural scene
        "In the gentle light of spring, a young Chinese woman in her 20s or 30s stands amidst a blooming garden, her hands gracefully positioned behind her back to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The flowers around her reflect the delicate blush of her cheeks, and her eyes, bright and curious, resemble the azure sky, clear and boundless. Each breeze that passes through the garden seems to whisper secrets to her, lifting strands of her hair in a dance with the petals that fall gently at her feet. The soft fabric of her dress flows like the gentle streams that meander through the fields, echoing the tranquility of nature itself. This serene photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman immersed in the natural beauty of spring, perfectly blending with the blooming garden.",
        "As summer arrives, the young Chinese woman in her 20s or 30s stands by a serene lake, her hands gently clasped together at her front to keep her fingers hidden, her upper body radiating warmth and vitality as she smiles brightly at the camera. The mountains and trees reflected in the still water behind her add a sense of peace and natural beauty to the scene. She wears comfortable, stylish outdoor attire that blends seamlessly with the surroundings. The air is fresh and crisp, enhancing the sense of tranquility. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the serene beauty of a mountain retreat.",
        "Autumn brings a tapestry of colors, and the young Chinese woman in her 20s or 30s walks through a forest adorned in hues of amber, gold, and crimson, her hands gently clasped together at her front to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The leaves crunch softly beneath her feet, a symphony of nature that accompanies her every step. Her eyes, now deeper and more contemplative, match the richness of the season. The cool air brushes against her skin, enhancing the rosy glow of her cheeks. Her presence in the autumnal landscape is a testament to the fleeting yet profound beauty of nature's transitions. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the rich tapestry of autumn.",
        "In the stillness of winter, the young Chinese woman in her 20s or 30s stands by a snow-covered pine tree, her hands gracefully positioned behind her back to keep her fingers hidden, her upper body reflecting the serene and elegant winter scene as she smiles warmly at the camera. She is clad in a warm, stylish coat, exuding quiet strength and dignified beauty. The snow-covered landscape creates a tranquil backdrop, with gentle flakes falling silently around her. She holds a woolen scarf close to her chest, the soft fabric adding to the cozy atmosphere. This photograph, focused on her upper body, captures the serene beauty and quiet strength of young Chinese women in the peaceful winter setting.",
        # city scene
        "In the heart of a bustling city, a young Chinese woman in her 20s or 30s stands on a quiet street corner, her hands gracefully positioned behind her back to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The tall skyscrapers and modern architecture rise behind her, their glass facades reflecting the soft morning light. She wears a stylish, urban outfit that complements the contemporary surroundings. The street is unusually empty, allowing her presence to dominate the scene. This photograph, focused on her upper body, captures the timeless beauty and sophisticated elegance of a young Chinese woman amidst the serene early morning of the city.",
        "As the sun sets over the city, the young Chinese woman in her 20s or 30s is found on a rooftop terrace, her hands resting gently on the terrace railing to keep her fingers hidden, her upper body radiating warmth and vitality as she smiles brightly at the camera. The cityscape stretches out behind her, with the evening lights beginning to twinkle like stars. Her outfit reflects a modern and chic style, perfectly suited for a night out in the city. The rooftop garden, with its lush greenery and twinkling fairy lights, frames her silhouette beautifully. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the vibrant energy of the city at dusk.",
        "Under the neon lights of the urban night, the young Chinese woman in her 20s or 30s stands confidently on a quiet street, her hands gently positioned behind her back to ensure her fingers are not prominent, her upper body exuding warmth and sophistication as she smiles gently at the camera. The colorful signs and lights illuminate her face, enhancing her features with a soft glow. She wears a fashionable, night-time outfit that highlights her elegance and style. The usually busy street is now calm, allowing her presence to shine against the backdrop of the vibrant city lights. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the dynamic and colorful atmosphere of the city at night.",
        "In a trendy urban café, a young Chinese woman in her 20s or 30s sits by the window, her hands gently resting on her lap to keep her fingers hidden, her upper body elegantly poised as she smiles warmly at the camera. The café is modern and stylish, with large windows letting in plenty of natural light. Her casual yet chic outfit blends seamlessly with the sophisticated décor of the café. The background shows a quiet street scene, devoid of other people, highlighting the tranquility of the moment. This photograph, focused on her upper body, captures the timeless beauty and grace of a young Chinese woman, harmonizing perfectly with the serene and contemporary atmosphere of the urban café.",
        # travel/resort scene
        "On a pristine beach, a young Chinese woman in her 20s or 30s stands with the ocean behind her, her hands gracefully positioned behind her back to ensure her fingers are not visible, her upper body gracefully poised and smiling warmly at the camera. The soft sand beneath her feet and the gentle waves crashing in the background create a serene and picturesque setting. She wears a light, flowy dress that complements the beach's natural colors. The sun sets behind her, casting a warm, golden glow that highlights her features. This photograph, focused on her upper body, captures the timeless beauty of a young Chinese woman, harmonizing perfectly with the tranquil and idyllic atmosphere of a beach at sunset.",
        "In the lush greenery of a mountain retreat, the young Chinese woman in her 20s or 30s stands by a serene lake, her hands gently clasped together at her front to keep her fingers hidden, her upper body radiating warmth and vitality as she smiles brightly at the camera. The mountains and trees reflected in the still water behind her add a sense of peace and natural beauty to the scene. She wears comfortable, stylish outdoor attire that blends seamlessly with the surroundings. The air is fresh and crisp, enhancing the sense of tranquility. This photograph, capturing her upper body, showcases the natural grace and elegance of young Chinese women, embodying the serene beauty of a mountain retreat.",
        "At a vibrant, tropical resort, the young Chinese woman in her 20s or 30s stands on a scenic beach path, her hands gracefully positioned behind her back to ensure her fingers are not visible, her upper body exuding warmth and sophistication as she smiles gently at the camera. The turquoise waves and the lush palm trees lining the path create a lively and luxurious backdrop. She wears a light, breezy dress that highlights her elegance and style. The clear sky and the sound of the ocean waves reflect the joyful and relaxing ambiance of the resort. This photograph, centered on her upper body, highlights the refined and serene aura of young Chinese women, perfectly blending with the dynamic and tropical atmosphere of the resort.",
        "In the peaceful setting of a serene lakeside park, a young Chinese woman in her 20s or 30s stands by the edge of the water, her hands gently clasped behind her back to keep her fingers hidden, her upper body gracefully poised as she smiles warmly at the camera. The calm lake and the surrounding trees create a picturesque and tranquil scene. She wears a light, summery dress that complements the natural colors of the park. The soft light of the late afternoon sun bathes her in a gentle glow, enhancing her features. This photograph, focused on her upper body, captures the timeless beauty and graceful presence of a young Chinese woman, harmonizing perfectly with the peaceful and idyllic atmosphere of the lakeside park.",
    ]

    ### actor
    prompts_actor = [
        # indoor scene
        "In the gentle light of spring, a young Chinese man in his 20s or 30s stands by a large window, his hands positioned confidently behind his back to keep his fingers hidden, his upper body displaying a strong and composed presence as he smiles warmly at the camera. The soft spring light filters through the curtains, casting a warm glow on his face and highlighting his determined expression. He wears a well-fitted, casual suit that complements the fresh, vibrant colors of the season. A vase of freshly picked flowers sits on the table beside him, adding a touch of nature to the scene. This photograph, focused on his upper body, captures the strong and charismatic aura of a young Chinese man, harmonizing perfectly with the tranquil and rejuvenating atmosphere of spring.",
        "As summer arrives, the young Chinese man in his 20s or 30s is found in a sun-drenched living room, his hands resting confidently on the armrest of a chair to keep his fingers hidden, his upper body exuding vitality and confidence as he smiles brightly at the camera. He sits comfortably on a sofa, wearing a light, breezy outfit that reflects the season's warmth. The large windows are open, allowing a gentle summer breeze to flow through, rustling the curtains and cooling the room. The vibrant colors of summer, with potted plants and bright cushions, frame his silhouette powerfully. This photograph, capturing his upper body, showcases the natural strength and confidence of young Chinese men, embodying the spirited energy of summer.",
        "Autumn brings a cozy, warm atmosphere as the young Chinese man in his 20s or 30s stands in a richly decorated living room, his hands positioned confidently behind his back to ensure his fingers are not prominent, his upper body exuding warmth and determination as he smiles gently at the camera. The room is adorned with autumnal decorations—dried leaves and warm-toned fabrics—reflecting the season's rich hues. The soft light of the late afternoon sun bathes him in a gentle glow, enhancing his features. This photograph, centered on his upper body, highlights the strong and assured presence of young Chinese men, perfectly blending with the cozy and introspective atmosphere of autumn.",
        "In the stillness of winter, the young Chinese man in his 20s or 30s stands by a frosted window, his hands confidently clasped together at his front to keep his fingers hidden, his upper body reflected in the glass and smiling warmly at the camera, revealing a calm and composed demeanor. He is clad in a warm, stylish coat, exuding quiet strength and dignified charm. The room is softly lit by the glow of nearby lamps, casting a gentle light that enhances the tranquility of the scene. Snow falls silently outside, adding to the peaceful ambiance. This photograph, focused on his upper body, captures the calm strength and quiet confidence of young Chinese men in the tranquil winter setting.",
        # natural scene
        "In the gentle light of spring, a young Chinese man in his 20s or 30s stands amidst a blooming garden, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and composure as he smiles warmly at the camera. The flowers around him reflect the fresh energy of the season, and his eyes, bright and focused, resemble the azure sky, clear and boundless. Each breeze that passes through the garden seems to invigorate him, lifting strands of his hair in a dance with the petals that fall gently at his feet. The soft fabric of his shirt flows like the gentle streams that meander through the fields, echoing the vitality of nature itself. This strong photograph, focused on his upper body, captures the timeless charm of a young Chinese man immersed in the natural beauty of spring, perfectly blending with the blooming garden.",
        "As summer arrives, the young Chinese man in his 20s or 30s stands by a serene lake, his hands confidently clasped together at his front to keep his fingers hidden, his upper body radiating warmth and vitality as he smiles brightly at the camera. The mountains and trees reflected in the still water behind him add a sense of peace and natural beauty to the scene. He wears comfortable, stylish outdoor attire that blends seamlessly with the surroundings. The air is fresh and crisp, enhancing the sense of tranquility. This photograph, capturing his upper body, showcases the natural strength and confidence of young Chinese men, embodying the serene beauty of a mountain retreat.",
        "Autumn brings a tapestry of colors, and the young Chinese man in his 20s or 30s walks through a forest adorned in hues of amber, gold, and crimson, his hands confidently clasped together at his front to ensure his fingers are not prominent, his upper body exuding warmth and determination as he smiles gently at the camera. The leaves crunch softly beneath his feet, a symphony of nature that accompanies his every step. His eyes, now deeper and more contemplative, match the richness of the season. The cool air brushes against his skin, enhancing the rugged charm of his features. His presence in the autumnal landscape is a testament to the fleeting yet profound beauty of nature's transitions. This photograph, centered on his upper body, highlights the strong and confident aura of young Chinese men, perfectly blending with the rich tapestry of autumn.",
        "In the stillness of winter, the young Chinese man in his 20s or 30s stands by a snow-covered pine tree, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting the serene and elegant winter scene as he smiles warmly at the camera. He is clad in a warm, stylish coat, exuding quiet strength and dignified charm. The snow-covered landscape creates a tranquil backdrop, with gentle flakes falling silently around him. This photograph, focused on his upper body, captures the calm strength and quiet confidence of young Chinese men in the peaceful winter setting.",
        # city scene
        "In the heart of a bustling city, a young Chinese man in his 20s or 30s stands on a quiet street corner, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and composure as he smiles warmly at the camera. The tall skyscrapers and modern architecture rise behind him, their glass facades reflecting the soft morning light. He wears a stylish, urban suit that complements the contemporary surroundings. The street is unusually empty, allowing his presence to dominate the scene. This photograph, focused on his upper body, captures the strong and charismatic aura of a young Chinese man amidst the serene early morning of the city.",
        "As the sun sets over the city, the young Chinese man in his 20s or 30s is found on a rooftop terrace, his hands resting confidently on the terrace railing to keep his fingers hidden, his upper body exuding vitality and determination as he smiles brightly at the camera. The cityscape stretches out behind him, with the evening lights beginning to twinkle like stars. His outfit reflects a modern and chic style, perfectly suited for a night out in the city. The rooftop garden, with its lush greenery and twinkling fairy lights, frames his silhouette powerfully. This photograph, capturing his upper body, showcases the natural strength and confidence of young Chinese men, embodying the vibrant energy of the city at dusk.",
        "Under the neon lights of the urban night, the young Chinese man in his 20s or 30s stands confidently on a quiet street, his hands confidently positioned behind his back to ensure his fingers are not prominent, his upper body exuding warmth and determination as he smiles gently at the camera. The colorful signs and lights illuminate his face, enhancing his features with a soft glow. He wears a fashionable, night-time outfit that highlights his strong and confident style. The usually busy street is now calm, allowing his presence to shine against the backdrop of the vibrant city lights. This photograph, centered on his upper body, highlights the strong and determined aura of young Chinese men, perfectly blending with the dynamic and colorful atmosphere of the city at night.",
        "In a trendy urban café, a young Chinese man in his 20s or 30s sits by the window, his hands confidently resting on his lap to keep his fingers hidden, his upper body exuding strength and composure as he smiles warmly at the camera. The café is modern and stylish, with large windows letting in plenty of natural light. His casual yet chic outfit blends seamlessly with the sophisticated décor of the café. The background shows a quiet street scene, devoid of other people, highlighting the tranquility of the moment. This photograph, focused on his upper body, captures the strong and composed charm of a young Chinese man, harmonizing perfectly with the serene and contemporary atmosphere of the urban café.",
        # travel/resort scene
        "On a pristine beach, a young Chinese man in his 20s or 30s stands with the ocean behind him, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and composure as he smiles warmly at the camera. The soft sand beneath his feet and the gentle waves crashing in the background create a serene and picturesque setting. He wears a light, flowy shirt that complements the beach's natural colors. The sun sets behind him, casting a warm, golden glow that highlights his features. This photograph, focused on his upper body, captures the timeless charm of a young Chinese man, harmonizing perfectly with the tranquil beauty of the beach.",
        "In a bustling market in a foreign land, a young Chinese man in his 20s or 30s stands amidst colorful stalls, his hands confidently positioned behind his back to keep his fingers hidden, his upper body exuding energy and curiosity as he smiles brightly at the camera. The vibrant array of goods and the dynamic atmosphere create a lively backdrop. He wears a casual, yet stylish outfit that blends seamlessly with the eclectic market scene. The mix of aromas from food stalls and the chatter of vendors enhance the sense of adventure. This photograph, capturing his upper body, showcases the natural vigor and exploratory spirit of young Chinese men, perfectly blending with the bustling market ambiance.",
        "In a tranquil mountain retreat, a young Chinese man in his 20s or 30s stands by a serene lake, his hands confidently clasped together at his front to keep his fingers hidden, his upper body reflecting the peaceful and majestic mountain landscape as he smiles gently at the camera. He is clad in a comfortable, stylish outdoor outfit, exuding quiet strength and a sense of adventure. The mountains and trees reflected in the still water behind him add a sense of calm and natural beauty to the scene. The crisp mountain air and the scent of pine trees enhance the tranquil atmosphere. This photograph, focused on his upper body, captures the natural grace and quiet strength of young Chinese men in the awe-inspiring mountain setting.",
        "At a scenic coastal village, the young Chinese man in his 20s or 30s stands by a charming harbor, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting the picturesque and serene environment as he smiles warmly at the camera. The quaint boats and colorful houses create a delightful and tranquil backdrop. He wears a light, casual outfit that suits the relaxed atmosphere of the village. The sound of gently lapping waves and the salty sea breeze add to the peaceful ambiance. This photograph, centered on his upper body, highlights the strong and calm presence of a young Chinese man, perfectly blending with the idyllic coastal village scenery.",
    ]

    prompts_actor_enhanced = [
        # indoor scene
        "In a stylish and modern apartment, a young Chinese man in his 20s or 30s stands by a large window, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body exuding strength and sophistication as he smiles warmly at the camera. The sleek furniture and contemporary decor create an elegant backdrop. He wears a crisp, tailored suit that enhances his sharp features and professional demeanor. The city skyline visible through the window adds a dynamic and cosmopolitan touch to the scene. This photograph, focused on his upper body, captures the refined and confident presence of a young Chinese man, perfectly blending with the modern and sophisticated atmosphere of the apartment.",
        "In the cozy warmth of a rustic library, the young Chinese man in his 20s or 30s stands next to a bookshelf filled with antique books, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting a quiet intellect and charm as he smiles gently at the camera. He is dressed in a smart-casual outfit, with a comfortable sweater and well-fitted jeans, blending seamlessly with the inviting ambiance of the library. The soft lighting casts a warm glow, highlighting the rich wood tones and creating a serene atmosphere. This photograph, focused on his upper body, showcases the intellectual and approachable charm of a young Chinese man, harmonizing perfectly with the cozy and scholarly environment of the library.",
        # natural scene
        "In the gentle light of spring, a young Chinese man in his 20s or 30s stands amidst a blooming garden, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and grace as he smiles warmly at the camera. The flowers around him reflect the fresh energy of the season, and the soft spring light enhances his features. He wears a light, casual shirt that complements the vibrant colors of the garden. The gentle breeze and the sweet scent of flowers add a sense of tranquility to the scene. This photograph, focused on his upper body, captures the timeless charm and strength of a young Chinese man, perfectly blending with the natural beauty of the spring garden.",
        "As summer arrives, the young Chinese man in his 20s or 30s stands by a serene lake, his hands confidently positioned behind his back to keep his fingers hidden, his upper body exuding vitality and composure as he smiles brightly at the camera. The mountains and trees reflected in the still water behind him add a sense of peace and natural beauty to the scene. He is clad in a light, breathable shirt and comfortable trousers, suitable for the summer weather. The clear blue sky and the sparkling water create a vibrant and lively backdrop. This photograph, capturing his upper body, showcases the natural strength and confident grace of a young Chinese man, harmonizing perfectly with the serene beauty of the lakeside.",
        # city scene
        "In the heart of a bustling city, a young Chinese man in his 20s or 30s stands on a quiet street corner, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and composure as he smiles warmly at the camera. The tall skyscrapers and modern architecture rise behind him, their glass facades reflecting the soft morning light. He wears a stylish, urban outfit that complements the contemporary surroundings. The street is unusually empty, allowing his presence to dominate the scene. This photograph, focused on his upper body, captures the serene and charismatic aura of a young Chinese man amidst the serene early morning of the city.",
        "As the sun sets over the city, the young Chinese man in his 20s or 30s is found on a rooftop terrace, his hands confidently positioned behind his back to keep his fingers hidden, his upper body exuding strength and elegance as he smiles brightly at the camera. The cityscape stretches out behind him, with the evening lights beginning to twinkle like stars. His outfit reflects a modern and chic style, perfectly suited for a night out in the city. The rooftop garden, with its lush greenery and twinkling fairy lights, frames his silhouette elegantly. This photograph, capturing his upper body, showcases the natural charisma and confident grace of a young Chinese man, embodying the vibrant energy of the city at dusk.",
        # travel/resort scene
        "On a pristine beach, a young Chinese man in his 20s or 30s stands with the ocean behind him, his hands confidently positioned behind his back to ensure his fingers are not visible, his upper body displaying strength and composure as he smiles warmly at the camera. The soft sand beneath his feet and the gentle waves crashing in the background create a serene and picturesque setting. He wears a light, casual shirt that complements the beach's natural colors. The sun sets behind him, casting a warm, golden glow that highlights his features. This photograph, focused on his upper body, captures the timeless charm and strength of a young Chinese man, harmonizing perfectly with the tranquil beauty of the beach.",
        "In a tranquil garden, the young Chinese man in his 20s or 30s stands beside a small pond, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting the peaceful and lush environment as he smiles gently at the camera. The garden, with its variety of vibrant flowers and neatly trimmed hedges, creates a serene and picturesque backdrop. He wears a comfortable, casual outfit that blends seamlessly with the natural beauty of the garden. The soft sound of water from a nearby fountain adds to the peaceful ambiance. This photograph, focused on his upper body, captures the calm and harmonious presence of a young Chinese man, perfectly complementing the tranquil garden setting.",
        "In a tranquil mountain retreat, a young Chinese man in his 20s or 30s stands by a serene lake, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting the peaceful and majestic mountain landscape as he smiles gently at the camera. He is clad in a comfortable, stylish outdoor outfit, exuding quiet strength and a sense of adventure. The mountains and trees reflected in the still water behind him add a sense of calm and natural beauty to the scene. The crisp mountain air and the scent of pine trees enhance the tranquil atmosphere. This photograph, focused on his upper body, captures the natural grace and quiet strength of young Chinese men in the awe-inspiring mountain setting.",
        "At a scenic coastal village, the young Chinese man in his 20s or 30s stands by a charming harbor, his hands confidently positioned behind his back to keep his fingers hidden, his upper body reflecting the picturesque and serene environment as he smiles warmly at the camera. The quaint boats and colorful houses create a delightful and tranquil backdrop. He wears a light, casual outfit that suits the relaxed atmosphere of the village. The sound of gently lapping waves and the salty sea breeze add to the peaceful ambiance. This photograph, centered on his upper body, highlights the strong and calm presence of a young Chinese man, perfectly blending with the idyllic coastal village scenery."
    ]

    ### businessman
    prompts_businessman = [
        "In a half-body shot, a young handsome Chinese male entrepreneur in his twenties stands confidently with his hands tucked behind his back, his determined gaze and poised smile highlighted by the serene backdrop of the Huangshan Mountains. The picturesque scenery, with its majestic peaks and misty valleys, symbolizes his ambitious spirit and innovative approach in China's evolving business landscape.",
        "In a half-body shot, a young handsome Chinese male entrepreneur in his twenties leans against a railing with his hands discreetly clasped behind him, his sharp eyes and confident expression framed by the serene waters of West Lake in Hangzhou. The tranquil lake, surrounded by lush greenery and ancient pagodas, underscores his strategic leadership and deep appreciation for cultural heritage.",
        "In a half-body shot, a young handsome Chinese male entrepreneur in his twenties stands with his hands casually tucked into the pockets of his lab coat, his focused expression and thoughtful eyes overseeing the development of groundbreaking inventions. The bustling innovation lab, filled with cutting-edge technology and prototypes, places him at the forefront of scientific innovation.",
        "In a half-body shot, a young handsome Chinese male entrepreneur in his twenties stands with his hands casually tucked behind him in a traditional tea garden, his serene expression and calm gaze reflecting his appreciation for cultural heritage. Ancient pagodas and serene gardens form the backdrop, highlighting his connection to tradition amidst his modern entrepreneurial pursuits.",
        "In a half-body shot, a young handsome Chinese male entrepreneur in his twenties stands confidently at a networking event, his hands discreetly in his pockets. His bright smile and poised demeanor are emphasized by the urban skyline behind him, underscoring his influence in shaping the business landscape of the metropolis.",
    ]

    ### businesswoman
    prompts_businesswoman = [
        "Gracefully poised, a successful young Chinese female entrepreneur in her twenties stands with a serene expression, her hands subtly tucked into the folds of her elegant coat as she overlooks the bustling streets of Paris. Her thoughtful gaze and confident smile reflect her sophisticated business acumen and global influence.",
        "In a boardroom filled with natural light, a beautiful young Chinese female entrepreneur in her twenties exudes confidence with her hands discreetly clasped behind her back, resting against a polished wooden table. Her composed expression and focused eyes convey her leadership in the tech industry.",
        "Amidst the tranquil setting of a classical art museum, a beautiful young Chinese female entrepreneur in her twenties stands contemplatively before a masterpiece, her hands gently folded behind her back. Her refined features and thoughtful expression highlight her appreciation for culture and her role as a patron of the arts.",
        "At a networking event in New York City, a beautiful young Chinese female entrepreneur in her twenties confidently engages with industry leaders, her hands elegantly gesturing while subtly tucked into her pockets. Her bright smile and poised demeanor underscore her pivotal role in shaping the business landscape of the metropolis.",
        "In a bustling innovation lab, a beautiful young Chinese female entrepreneur stands with her hands casually tucked into the pockets of her lab coat. Surrounded by cutting-edge technology and prototypes, she oversees the development of groundbreaking inventions, her focused expression highlighting her role at the forefront of scientific innovation.",
    ]

    ### male scientist
    prompts_male_scientist = [
        "In a sleek laboratory setting, a young Chinese scientist, focused and intense, adjusts equipment with precise gestures. His upper body is clad in a crisp white lab coat, concealing hands that expertly manipulate unseen tools within hidden pockets. His gaze, subtly directed towards the camera, exudes a quiet confidence in his work.",
        "Amidst a bustling research facility, a Chinese researcher in his twenties meticulously examines a specimen, his torso framed by a cluttered lab bench. His hands, mysteriously tucked away, suggest a concealed mastery of scientific instruments. His eyes, momentarily lifting to meet the camera, convey both concentration and a hint of curiosity.",
        "Outdoors in a field station, a young Chinese scientist stands amidst experimental equipment, his upper body clad in a practical field jacket. Hands discreetly tucked, perhaps in pockets or behind his back, hint at a thoughtful demeanor as he gazes towards the camera, embodying a blend of scientific inquiry and natural surroundings.",
        "Engaged in a futuristic laboratory, a Chinese researcher, aged around thirty, studies data on a holographic display. His upper body, enveloped in a high-tech lab suit, obscures any visible hand movements, suggesting a seamless integration with advanced technology. His gaze subtly shifts towards the camera, reflecting a dedication to cutting-edge research.",
        "Against the backdrop of a bustling research conference, a young Chinese scientist presents findings with a poised upper body, hands subtly concealed. His expression, as he addresses the audience or camera, conveys both scholarly enthusiasm and a composed focus on sharing scientific insights.",
        "In a serene observatory nestled among mountains, a young Chinese astrophysicist gazes skyward, his upper body framed against the glowing cosmos. His hands, discreetly out of sight, suggest a deep contemplation of celestial mysteries as his eyes subtly meet the camera, reflecting a blend of scientific wonder and personal reverence for the universe.",
        "Inside a bustling biotech startup, a Chinese researcher in his twenties navigates a lab teeming with innovative prototypes. His upper body, clad in a modern lab coat, conceals hands engaged in precise genetic editing. His gaze, momentarily directed towards the camera, exudes a mix of ambition and quiet determination in pioneering scientific advancements.",
        "Amidst an archaeological dig in a remote Chinese province, a young scientist carefully brushes dirt from ancient artifacts. His upper body, wrapped in weathered field gear, masks hands expertly maneuvering delicate tools. His eyes, focused on the excavation site or camera, convey a profound respect for history and a dedication to uncovering ancient secrets.",
        "Within a vibrant robotics lab in Shanghai, a Chinese engineer in his thirties oversees the assembly of a humanoid prototype. His upper body, adorned with a lab coat bearing technological insignias, conceals hands deftly programming intricate circuitry. His gaze, briefly meeting the camera, reflects a blend of technical mastery and visionary ambition in artificial intelligence.",
        "In a bustling urban setting, a young Chinese environmental scientist conducts field measurements amidst a panorama of skyscrapers. His upper body, adorned with practical field gear, masks hands engaged in sampling urban air quality. His gaze, directed towards the camera amid the cityscape, embodies a blend of scientific rigor and a commitment to improving urban sustainability.",
    ]

    ### female scientist
    prompts_female_scientist = [
        "Directly facing the camera in a modern genetics lab, a young Chinese female scientist stands confidently, her hands discreetly tucked into the pockets of her pristine lab coat. Her features are illuminated by the soft glow of laboratory lights, framed by dark hair neatly pulled back. Her expression, a blend of curiosity and determination, reflects her deep commitment to unraveling the mysteries of DNA through rigorous research.",
        "In a state-of-the-art neuroscience research institute, a Chinese neuroscientist in her twenties stands before brain imaging monitors, her hands expertly manipulating controls hidden behind her back. Her face is centered and composed for the camera, her expression tinged with curiosity and resolve as she delves into the intricacies of neural networks. Dressed in a professional lab coat, her direct gaze embodies her dedication to understanding the complexities of the human brain.",
        "Beside a humanoid robot prototype in a dynamic robotics lab, a Chinese engineer specializing in AI stands confidently facing the camera, her hands subtly adjusting components concealed within her sleek lab suit. Her features are sharp and focused under the glow of computer screens, reflecting both her technical expertise and visionary leadership in advancing robotics technology.",
        "Within a sleek biotech laboratory in Shanghai, a young Chinese female scientist stands with poise, her hands subtly adjusting equipment hidden behind her back. Her face, softly illuminated by the ambient light of the lab, reveals a thoughtful expression framed by dark hair swept elegantly to one side. Dressed in a tailored lab coat adorned with subtle scientific motifs, her focused gaze towards the camera conveys both intellect and determination in the pursuit of groundbreaking genetic research.",
        "In the midst of a bustling robotics research facility in Beijing, a Chinese engineer specializing in AI stands confidently beside a cutting-edge robot prototype. Her hands, skillfully concealed within the pockets of her lab coat, hint at expertise in programming and design. With her face illuminated by the glow of digital displays, her expression reflects a blend of innovation and meticulous attention to detail, embodying the future-oriented spirit of technological advancement.",
        "At an atmospheric chemistry research center in Guangzhou, a young Chinese atmospheric scientist stands before a bank of monitoring instruments, her hands discreetly holding a sample collector behind her back. Her face, bathed in the soft glow of the lab's environmental lighting, exudes a quiet intensity and focus. Dressed in specialized field attire, her direct gaze towards the camera underscores her commitment to understanding and mitigating environmental challenges through rigorous scientific study.",
        "Within a serene botanical research station in Yunnan, a Chinese botanist in her twenties carefully examines plant specimens, her hands subtly hidden within the pockets of her field jacket. Her face, lit by natural sunlight filtering through the greenhouse, reflects a deep connection to nature and a meticulous approach to botanical study. Dressed in practical field gear, her focused gaze towards the camera captures both her scientific passion and dedication to preserving biodiversity.",
        "Positioned in a high-tech genetics lab, a young Chinese female scientist stands confidently, her hands deftly manipulating a DNA sequencing machine concealed behind her back. Her face, illuminated by the soft glow of laboratory lights, displays a focused expression framed by dark hair swept back neatly. Dressed in a pristine lab coat adorned with intricate genetic patterns, her direct gaze towards the camera conveys both intellectual prowess and a determined pursuit of scientific discovery.",
        "Within a dynamic neuroscience research institute, a Chinese neuroscientist in her twenties stands before an array of brain imaging monitors, her hands subtly adjusting settings hidden behind her tailored lab coat. Her face, composed and thoughtful under the ambient lab lighting, reflects a deep engagement with the complexities of neural networks. Her attire, a blend of professional elegance and scientific rigor, complements her direct gaze towards the camera, symbolizing her dedication to unraveling the mysteries of the human brain.",
        "Beside a prototype of an advanced humanoid robot in a bustling robotics lab, a Chinese engineer specializing in AI stands with confidence, her hands subtly adjusting circuits concealed within her sleek lab suit. Her face, illuminated by the digital displays around her, exudes a blend of innovation and precision. Dressed in cutting-edge attire that merges technology with style, her focused gaze towards the camera embodies both technical expertise and visionary leadership in shaping the future of robotics.",
    ]

    ### sportsman
    prompts_sportsman = [
        "A male Chinese athlete is shown from the waist up, facing the camera directly with his hands in his pockets. He has short black hair and a fit, muscular build. The background features a variety of popular Chinese sports such as basketball, table tennis, badminton, gymnastics, and martial arts.",
        "From the waist up, a male Chinese athlete faces the camera with his hands behind his back. He has short black hair, a strong physique, and is dressed in sportswear. The background includes scenes depicting diverse Chinese sports activities, including basketball, table tennis, badminton, gymnastics, and taekwondo.",
        "The image shows a male Chinese athlete from the waist up, facing the camera with his hands in his pockets. He has short black hair, a muscular build, and is wearing athletic clothing. The background features various popular Chinese sports like basketball, table tennis, badminton, gymnastics, and swimming.",
        "A male Chinese athlete is depicted from the waist up, with his hands behind his back, facing the camera. He showcases short black hair and a fit build. The background highlights a range of popular Chinese sports, including basketball, table tennis, badminton, gymnastics, and volleyball.",
        "A male Chinese athlete is captured from the chest up, facing directly towards the camera with his hands clasped behind his back. His short black hair and strong, muscular physique are prominent. The background showcases a variety of popular Chinese sports, including basketball, table tennis, badminton, gymnastics, and martial arts.",
    ]

    ### sportswoman
    prompts_sportswoman = [
        # indoor
        "A beautiful Chinese female athlete is depicted from the waist up, facing the camera with her hands in her pockets. She is smiling brightly after just finishing a badminton match. An indoor sports hall with badminton courts can be seen behind her.",
        "From the chest up, a beautiful Chinese female athlete faces the camera confidently, with her hands behind her back. She is beaming with joy after a successful dance performance. Behind her, an indoor dance studio with mirrors and ballet bars is visible.",
        "The image captures a beautiful Chinese female athlete from the waist up, facing the camera with her hands in her pockets. She smiles softly, having just completed a yoga session. The background shows an indoor gym with yoga mats and serene lighting.",
        "A beautiful Chinese female athlete is shown from the chest up, facing the camera with her hands in her pockets. She is glowing with happiness after a swimming competition. An indoor pool with lanes and starting blocks is seen behind her.",
        # outdoor
        "A beautiful Chinese female athlete is depicted from the waist up, facing the camera with her hands in her pockets. She is smiling brightly after just finishing a tennis match. A tennis court with green surroundings can be seen behind her.",
        "From the chest up, a beautiful Chinese female athlete faces the camera confidently, with her hands behind her back. She is beaming with joy after a successful soccer game. Behind her, an outdoor soccer field with lush grass and goalposts is visible.",
        "The image captures a beautiful Chinese female athlete from the waist up, facing the camera with her hands in her pockets. She smiles softly, having just completed a track and field event. The background shows an outdoor track with a stadium and blue sky.",
        "A beautiful Chinese female athlete is shown from the chest up, facing the camera with her hands in her pockets. She is glowing with happiness after a game of ultimate frisbee. An outdoor field with clear skies and distant trees is seen behind her.",
        "From the waist up, a beautiful Chinese female athlete faces the camera with her hands behind her back. She smiles brightly, having just finished a basketball game. The background features an outdoor basketball court with a hoop and urban skyline visible.",
    ]

    ### male educator
    prompts_male_educator = [
        "A young Chinese male educator in his 20s is smiling with his face prominently displayed. His hands are placed behind his back, and the photo captures his upper body. The background features a blackboard filled with mathematical formulas, emphasizing his knowledge and authority.",
        "In a modern university lecture hall, a young Chinese male educator in his 20s stands with his face prominently displayed, looking towards the camera. His hands are behind his back, and the photo captures his upper body. The background shows a projector screen with a complex graph, highlighting his expertise.",
        "In front of a large bookshelf filled with books, a young Chinese male educator in his 20s stands with his face prominently displayed. His hands are behind his back, and the photo captures his upper body. Dressed in a comfortable sweater and shirt, he exudes passion for education and learning.",
        "On a sunny day, a young Chinese male educator in his 20s stands prominently in front of a prestigious university building. His hands are behind his back, and the photo captures his upper body. Wearing a stylish blazer, he appears vibrant and engaging.",
        "In a high-tech classroom, a young Chinese male educator in his 20s stands with his face prominently displayed. His hands are behind his back, and the photo captures his upper body. Dressed in a sleek, modern suit, he represents a forward-thinking approach to teaching.",
    ]

    ### female educator
    prompts_female_educator = [
        # hands behind back
        "A young Chinese female educator in her 20s stands with a graceful smile, her face prominently displayed. Her hands are placed behind her back, and the photo captures her upper body. She stands in front of a bookshelf filled with academic books, radiating elegance and intelligence.",
        "In a modern classroom, a beautiful Chinese female educator in her 20s stands confidently with her face prominently displayed. Her hands are behind her back, and the photo captures her upper body. The background shows a smartboard with educational content, highlighting her intellectual demeanor.",
        "A young Chinese female educator in her 20s stands outdoors in a serene university courtyard, her face prominently displayed. Her hands are behind her back, and the photo captures her upper body. She is dressed in a chic, professional outfit, exuding both elegance and intellect.",
        "In a high-tech lecture hall, a beautiful Chinese female educator in her 20s stands with her face prominently displayed, looking towards the camera. Her hands are behind her back, and the photo captures her upper body. The background features a digital screen with complex equations, showcasing her academic prowess.",
        "A young Chinese female educator in her 20s stands in front of a large window overlooking a lush garden, her face prominently displayed. Her hands are behind her back, and the photo captures her upper body. Wearing a stylish yet professional dress, she embodies both beauty and intelligence."
        # hands into pockets
        "A young Chinese female educator in her 20s stands elegantly with a warm smile, her face prominently displayed. Her hands rest comfortably in her pockets, and the photo captures her upper body. She stands in front of a bookshelf filled with academic books, exuding both intelligence and sophistication.",
        "In a modern classroom, a beautiful Chinese female educator in her 20s stands confidently with her face prominently displayed. Her hands are tucked into her pockets, and the photo captures her upper body. The background shows a smartboard filled with educational diagrams, highlighting her intellectual and poised demeanor.",
        "A young Chinese female educator in her 20s stands outdoors in a picturesque university courtyard, her face prominently displayed. Her hands are placed casually in her pockets, and the photo captures her upper body. She is dressed in a stylish, professional outfit that radiates elegance and intellect.",
        "In a state-of-the-art lecture hall, a beautiful Chinese female educator in her 20s stands with her face prominently displayed, her expression serene and confident. Her hands rest in her pockets, and the photo captures her upper body. The background features a digital screen with detailed charts and graphs, showcasing her academic excellence.",
        "A young Chinese female educator in her 20s stands by a large window overlooking a tranquil garden, her face prominently displayed. Her hands are placed in her pockets, and the photo captures her upper body. Wearing a chic and professional dress, she embodies both beauty and intelligence, reflecting her dedication to education.",
    ]

    create_dirs(dir_name="./assets_sd3")

    # init inference runner
    generator = None
    if model == "sdxl":
        # use sdxl
        generator = ExpertGenerator()
        generator.init(improve=False)
        generator.adjust_steps(steps=100, ratio=0.7)
    elif model == "sd3":
        # use sd3-medium model
        generator = SD3Generator()
        generator.init()
        generator.adjust_steps(steps=50)
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
        prompts = prompts_male_educator
        value = 7.0
        start = 0  # this needs to be changed accordingly
        for i in range(start, start + count):
            id = random.randint(0, len(prompts) - 1)    # generate random number
            # id = 0
            # output_file = os.path.join('./output', f'{i}M.jpg')
            output_file = os.path.join('./output', f'{i}F.jpg')
            generator.generate_template(prompt=prompts[id], negative_prompt=negative_prompt, output_file=output_file, guidance_scale=value)


if __name__ == '__main__':
    rename("./output", "M", start=0)
    # demo(100, model="sd3")
    # demo(150, model="sdxl")
    # batch_generation(100)
