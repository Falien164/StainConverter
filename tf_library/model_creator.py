from tf_library.models.pix2pix import Pix2Pix
from tf_library.models.cyclegan import CycleGAN
from tf_library.models.unet_generator import UnetGenerator
from tf_library.models.resnet_generator import ResnetGenerator
from tf_library.models.patchgan_discriminator import PatchGAN


class ModelCreator:
    @staticmethod
    def create_model(type_of_gan: str):
        type_of_gan = type_of_gan.lower()

        if type_of_gan == "pix2pix":
            generator = UnetGenerator()()
            discriminator = PatchGAN()(type_of_gan=type_of_gan, size='30x30')
            return Pix2Pix(generator, discriminator)

        elif type_of_gan == "pix2pix_with_resnet_gen":
            generator = ResnetGenerator()()
            discriminator = PatchGAN()(type_of_gan=type_of_gan, size='16x16')
            return Pix2Pix(generator, discriminator)

        elif type_of_gan == "cyclegan":
            generator = ResnetGenerator()()
            discriminator = PatchGAN()(type_of_gan=type_of_gan, size='16x16')
            return CycleGAN(generator, discriminator)

        elif type_of_gan == "cyclegan_with_unet_gen":
            generator = UnetGenerator()()
            discriminator = PatchGAN()(type_of_gan=type_of_gan, size='30x30')
            return CycleGAN(generator, discriminator)
        else:
            raise "Such model cannot be created. Check env file."
