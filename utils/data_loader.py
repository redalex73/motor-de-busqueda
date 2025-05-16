import os

def get_files_by_extension(data_dir, extensions):
    return [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(extensions)
    ]

def load_data_paths(data_dir):
    image_paths = get_files_by_extension(data_dir, ('.jpg', '.jpeg', '.png'))
    text_paths = get_files_by_extension(data_dir, ('.txt',))
    return image_paths, text_paths
