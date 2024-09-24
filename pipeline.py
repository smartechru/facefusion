import os
import time
import string
import random
import concurrent.futures
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
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=n))
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
    career_list = ["artist", "scientist", "business", "educator", "sports"]
    if career in career_list:
        # set photo template according to the given gender
        num = random.randint(0, max_num - 1)  # set random style
        dst = f"./assets/{career}/{num}{sex}.jpg"  # set target image path

        # set output image path
        os.makedirs("./output", exist_ok=True)
        output = f"./output/{generate_name()}.jpg"
        process_image(src, dst, output)
        print(f"Output image: {output}")
    else:
        print("Career name incorrect!")


def process_image(src: str, dst: str, out: str) -> None:
    """
    :param src: source image path
    :param dst: target image path
    :param out: output image path
    :return: none
    """
    # set essential parameters
    output_image_resolution = detect_image_resolution(dst)
    facefusion.globals.output_image_resolution = pack_resolution(
        output_image_resolution
    )
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
    print("[PIPELINE] System Initialized!")
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    # dummy test
    dir_path = "D:/AI\Dataset/Celebrity Recognition DB_65/Bai Lu_004206_F"
    files = os.listdir(dir_path)
    for file in files:
        print(f"[PIPELINE] Submit for {file}")
        src_path = os.path.join(dir_path, file)
        num = random.randint(0, 281)
        dst_path = f"D:/AI/Engine/engine/faceswap/assets/arts/{num}F.jpg"
        output = f"./output/{generate_name()}.jpg"
        pool.submit(process_image, src_path, dst_path, output)

    # shutdown task
    pool.shutdown(wait=True)
    print("{PIPELINE] Done!")


if __name__ == "__main__":
    init_system()
