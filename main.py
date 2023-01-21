from datetime import datetime

import os
import environ
import sys

from model_creator import ModelCreator
from utils import generate_image


class StainConverter:
    def __init__(self):
        self.results_filename = None
        self.best_fid = sys.float_info.max

    def training_pipeline(self):
        self.create_folder_for_results()

        train_he_images_path = os.environ.get('TRAIN_HE_IMAGES_PATH')
        train_pas_images_path = os.environ.get('TRAIN_PAS_IMAGES_PATH')
        test_he_images_path = os.environ.get('TEST_HE_IMAGES_PATH')
        test_pas_images_path = os.environ.get('TEST_PAS_IMAGES_PATH')
        # TODO: Validate if paths are correct

        data_loader = DataLoader(train_he_images_path=train_he_images_path, train_pas_images_path=train_pas_images_path,
                                 test_he_images_path=test_he_images_path, test_pas_images_path=test_pas_images_path)
        train_dataset, test_dataset = data_loader.load_dataset()

        data_loader.display_sample_pair(train_dataset, self.results_filename)

        train_dataset = train_dataset.take(3)
        test_dataset = test_dataset.take(2)

        model = ModelCreator.create_model(library=library, type_of_gan=os.environ.get('MODEL'))

        self.train(model, train_dataset, test_dataset)

        self.load_model(model.generator)

        self.generate_transformed_images(model, test_dataset)

        # self.calculate_metrics()
        # TODO: calculate final metrics

    def train(self, model, train_dataset, test_dataset):
        for step, (input_img, target_img) in train_dataset.repeat().take(int(os.environ['N_STEPS'])).enumerate():
            model.train_step(input_img, target_img)
            fid = calculate_fid(model.generator, test_dataset)
            self.save_fid(fid, step)
            if fid < self.best_fid:
                self.save_model(model.generator)
                self.best_fid = fid

    def create_folder_for_results(self):
        starting_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.results_filename = os.path.join(os.getcwd(), os.environ['RESULTS_FOLDER'],
                                             os.environ['MODEL'] + starting_datetime) + '/'
        os.mkdir(self.results_filename)
        os.mkdir(os.path.join(self.results_filename, 'generated_images'))

    def generate_transformed_images(self, model, test_dataset):
        for idx, (input_img, target_img) in enumerate(test_dataset.take(len(test_dataset))):
            generate_image(model.generator, input_img, target_img, self.results_filename + "/generated_images/", idx)

    def save_fid(self, fid, step):
        with open(self.results_filename + 'fid.txt', 'a') as file:
            file.write(f"FID in step {step} = {fid} \n")

    def save_model(self, model):
        model.save_weights(self.results_filename + 'gen.h5')
        # TODO: save all models

    def load_model(self, model):
        model.load_weights(self.results_filename + 'gen.h5')
        # TODO: load all models


if __name__ == "__main__":
    env_path = os.path.join(os.getcwd(), "_config", ".env")
    if os.path.exists(env_path):
        environ.Env().read_env(env_path)
        print(".env found and loaded")
    else:
        print(".env not found")
        exit()
    library = os.environ.get('LIBRARY', '').lower()

    if library == 'tensorflow':
        from tf_library.data_loader import DataLoader
        from tf_library.metrics.fid import calculate_fid
    elif library == "torch":
        from torch_library.data_loader import DataLoader
        from torch_library.metrics import fid
        # TODO: code models in PyTorch
    else:
        raise Exception("set env variable LIBRARY to 'tensorflow' or 'torch'")

    stain_converter = StainConverter()
    stain_converter.training_pipeline()
