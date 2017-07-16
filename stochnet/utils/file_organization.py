import os


def create_dir_if_it_does_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_dataset_folder(data_root_folder, timestep, dataset_id):
    dataset_folder = os.path.join(data_root_folder, str(timestep))
    dataset_folder = os.path.join(dataset_folder, str(dataset_id))
    create_dir_if_it_does_not_exist(dataset_folder)
    return dataset_folder
