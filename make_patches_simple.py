import os

import openslide
from tqdm import tqdm

svs_dir = './data/raw_data/TCGA_Stomach_452'
save_dir = './data/patch_data/TCGA_Stomach_452_simple'

read_size = 1024
step = 1.0

if __name__ == '__main__':
    # 1. Get SVS Paths
    svs_paths = {}
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext != '.svs':
                continue
            file_index = filename.strip(ext)
            svs_path = os.path.join(path, filename)
            svs_paths[file_index] = svs_path

    # 2. Make Patches
    for file_index, svs_path in tqdm(svs_paths.items()):
        slide = openslide.OpenSlide(svs_path)
        w_pixels, h_pixels = slide.level_dimensions[0]

        patch_save_dir = os.path.join(save_dir, file_index)
        os.makedirs(patch_save_dir, exist_ok=True)
        for w_i in range(0, w_pixels, int(read_size * step)):
            for h_i in range(0, h_pixels, int(read_size * step)):
                slide_img = slide.read_region((w_i, h_i), 0, (read_size, read_size))
                save_path = os.path.join(patch_save_dir, '{}_patch_{}_{}.png'.format(file_index, w_i, h_i))
                slide_img.save(save_path)
