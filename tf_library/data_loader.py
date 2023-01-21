import tensorflow as tf
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self, train_he_images_path, train_pas_images_path, test_he_images_path, test_pas_images_path,
                 img_height=256, img_width=256, buffer_size=400, batch_size=1, seed=1234):
        self.train_he_images_path = train_he_images_path
        self.train_pas_images_path = train_pas_images_path
        self.test_he_images_path = test_he_images_path
        self.test_pas_images_path = test_pas_images_path
        self.img_height = img_height
        self.img_width = img_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = tf.random.set_seed(seed=seed)

    def load_dataset(self):
        train_dataset = tf.data.Dataset.list_files(self.train_he_images_path + '*.png')
        train_dataset = train_dataset.map(self._load_image_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.buffer_size, seed=self.seed)
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.list_files(self.test_he_images_path + '*.png')
        test_dataset = test_dataset.map(self._load_image_dataset)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset

    def _load_pair_images(self, he_image_filepath: str):
        he_image = self._read_image(he_image_filepath)
        pas_image_filepath = tf.strings.regex_replace(he_image_filepath, "HE", "PAS")
        pas_image = self._read_image(pas_image_filepath)
        he_image, pas_image = self._resize(he_image, pas_image, self.img_height, self.img_width)
        he_image = tf.cast(he_image, tf.float32)
        pas_image = tf.cast(pas_image, tf.float32)
        return he_image, pas_image

    def _load_image_dataset(self, image_file: str):
        he_image, pas_image = self._load_pair_images(image_file)
        he_image, pas_image = self._resize(he_image, pas_image, self.img_height, self.img_width)
        he_image, pas_image = self._normalize(he_image, pas_image)
        return he_image, pas_image

    @staticmethod
    def display_sample_pair(dataset, path):
        images = next(iter(dataset.take(1)))
        plt.figure()
        for idx, img in enumerate(images):
            plt.subplot(1, 2, idx + 1)
            plt.title(f"Domain {idx}")
            plt.imshow(img[0] * 0.5 + 0.5)  # by default is (1, 256, 256, 3) in pixel range (-1, 1)
            plt.axis('off')
        path = os.path.join(path, 'sample.png')
        plt.savefig(path)

    @staticmethod
    def _read_image(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, dtype=tf.uint8)
        return image[:, :, 0:3]

    @staticmethod
    def _resize(he_image, pas_image, height, width):
        he_image = tf.image.resize(he_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        pas_image = tf.image.resize(pas_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return he_image, pas_image

    @staticmethod
    def _normalize(he_image, pas_image):
        """
        Normalizing the images to [-1, 1]
        """
        he_image = (he_image / 127.5) - 1
        pas_image = (pas_image / 127.5) - 1

        return he_image, pas_image
