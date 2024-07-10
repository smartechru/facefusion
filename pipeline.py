import os
import time
import string
import random
import facefusion.globals
from facefusion import core
from facefusion.vision import detect_image_resolution, pack_resolution
from facefusion.face_store import clear_static_faces, clear_reference_faces


def generate_name(n: int = 7) -> str:
    """
    Generate a random name with upper case letters
    :param n: size of string
    :return: random name
    """
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))
    return str(res)


def run(src: str, career: str, sex: str, max_num: int = 10) -> None:
    """
    :param src: source image path
    :param career: career cluster name
    :param sex: male or female { 'M', 'F' }
    :param max_num: maximum number of templates
    :return: none
    """
    # verify input parameter
    career_list = [ 'artist', 'scientist', 'business', 'educator', 'sports' ]
    if career in career_list:
        # set photo template according to the given gender
        num = random.randint(0, max_num - 1)        # set random style
        dst = f'./assets/{career}/{num}{sex}.jpg'   # set target image path

        # set output image path
        os.makedirs('./output', exist_ok=True)
        output = f'./output/{generate_name()}.jpg'
        process_image(src, dst, output)
        print(f'Output image: {output}')
    else:
        print('Career name incorrect!')


def process_image(src: str, dst: str, out: str) -> None:
    """
    :param src: source image path
    :param dst: target image path
    :param out: output image path
    :return: none
    """
    # set essential parameters
    output_image_resolution = detect_image_resolution(dst)
    facefusion.globals.output_image_resolution = pack_resolution(output_image_resolution)
    facefusion.globals.target_path = dst
    facefusion.globals.output_path = out
    facefusion.globals.source_paths = [src]

    # run headless process
    clear_reference_faces()
    clear_static_faces()
    core.conditional_process()


def init_system() -> None:
    # initialize system
    core.cli()
    print("[FACEFUSION.MAIN] System Initialized!")

    # dummy test
    src_1 = "test/pair_1_man.jpg_pair_1_woman.jpg_father_0.7_female.png_20.png"
    src_2 = "test/man.jpg_woman.jpg_father_0.5_male_with_optimization.png"
    src_3 = "test/man.jpg_woman.jpg_father_0.5_male_without_optimization.png"
    src_4 = "D:/AI/BabyGen/BabyGAN/data/images/baby_pair_4_man.jpg_pair_4_woman.jpg_father_0.9_female.png_50.png"
    src_5 = "D:/AI/BabyGen/BabyGAN/data/images/baby_pair_4_man.jpg_pair_4_woman.jpg_mother_0.9_female.png_50.png"
    src_6 = "D:/AI/BabyGen/BabyGAN/data/images/pair_4_man.jpg_pair_4_woman.jpg_father_0.9_female.png"
    src_7 = "D:/AI/BabyGen/BabyGAN/data/images/pair_4_man.jpg_pair_4_woman.jpg_father_0.9_male.png"
    src_8 = "D:/AI/BabyGen/BabyGAN/data/images/pair_4_man.jpg_pair_4_woman.jpg_mother_0.9_female.png"
    src_9 = "D:/AI/BabyGen/BabyGAN/data/images/pair_4_man.jpg_pair_4_woman.jpg_mother_0.9_male.png"
    src_10 = "D:/AI/BabyGen/BabyGAN/data/images/baby_pair_1_man.jpg_pair_1_woman.jpg_father_0.9_male.png_50.png"
    
    career = 'artist'
    sex = 'F'
    src = src_10
    
    # for i in range(0, 10):
        # run(src, career, sex, 100)
        # run(src_3, career, sex, 44)
    
    # dst = f'./output/normal/{i}F.jpg'
    dst = f'./assets/artist/5M.jpg'
    output = f'./output/{generate_name()}.jpg'
    output = f'./output/2.jpg'
    process_image(src, dst, output)


if __name__ == "__main__":
    init_system()
