from torch_library.models.pix2pix import Pix2Pix
from torch_library.models.cyclegan import CycleGAN
from torch_library.models.unet_generator import UnetGenerator
from torch_library.models.resnet_generator import ResnetGenerator
from torch_library.models.discriminator import Discriminator


class ModelCreator:
    @staticmethod
    def create_model(type_of_gan: str):
        type_of_gan = type_of_gan.lower()
        input_n_ch = 3
        output_n_ch = 3

        if type_of_gan == "pix2pix":
            generator = UnetGenerator(input_n_ch, output_n_ch)
            discriminator = Discriminator(input_n_ch * 2)
            return Pix2Pix(generator, discriminator)

        elif type_of_gan == "pix2pix_with_resnet_gen":
            generator = ResnetGenerator(input_n_ch, output_n_ch)
            discriminator = Discriminator(input_n_ch * 2)
            return Pix2Pix(generator, discriminator)

        elif type_of_gan == "cyclegan":
            generator = ResnetGenerator(input_n_ch, output_n_ch)
            discriminator = Discriminator(input_n_ch)
            return CycleGAN(generator, discriminator)

        elif type_of_gan == "cyclegan_with_unet_gen":
            generator = UnetGenerator(input_n_ch, output_n_ch)
            discriminator = Discriminator(input_n_ch)
            return CycleGAN(generator, discriminator)
        else:
            raise "Such model cannot be created. Check env file."
