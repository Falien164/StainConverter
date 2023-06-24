from datetime import datetime

import os
import environ
import sys

from image_metrics import ImageMetrics


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

        model = ModelCreator.create_model(type_of_gan=os.environ.get('MODEL'))

        self.trainer = Trainer(self.results_filename)
        self.trainer.train(model, train_dataset, test_dataset)

        # self.load_best_model(model.generator)

        self.trainer.generate_comparative_images(model, test_dataset)

        self.calculate_metrics(model, test_dataset)

    def create_folder_for_results(self):
        starting_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.results_filename = os.path.join(os.getcwd(), os.environ['RESULTS_FOLDER'],
                                             os.environ['MODEL'] + starting_datetime) + '/'
        os.mkdir(self.results_filename)
        os.mkdir(os.path.join(self.results_filename, 'generated_images'))

    def calculate_metrics(self, model, test_dataset):
        metrics_handler = ImageMetrics(len(test_dataset))
        library = os.environ.get('LIBRARY', '').lower()
        if library == "tensorflow":
            results = metrics_handler.count_metrics_for_tensorflow(model, test_dataset)
        if library == "torch":
            results = metrics_handler.count_metrics_for_torch(model, test_dataset)
        metrics_path = self.results_filename + os.environ["METRICS_FILENAME"]
        metrics_handler.export_metrics_to_file(metrics=results, metrics_result_filename=metrics_path)

    def load_best_model(self, model):
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
        from tf_library.model_creator import ModelCreator
        from tf_library.trainer import Trainer
    elif library == "torch":
        from torch_library.data_loader import DataLoader
        from torch_library.model_creator import ModelCreator
        from torch_library.trainer import Trainer
        # TODO: code models in PyTorch
    else:
        raise Exception("set env variable LIBRARY to 'tensorflow' or 'torch'")

    stain_converter = StainConverter()
    stain_converter.training_pipeline()
