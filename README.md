# StainConverter
The scientific goal of this project is to use GAN to generate an artifical images of kidney's glomeruli from one stain into another (HE to PAS). For more information read one of the following papers (cited on a bottom of the page):
- HE-to-PAS histological stain conversion by GAN in renal pathology (IEEE Research paper)
- GAN application in the histological stain conversion (Master's thesis, Warsaw University of Technology)

The secondary goal of this project is to compare usage of a PyTorch and TensorFlow libraries. Therefore, a training pipeline, which can handle each library separatly,  was created. It can call either TensorFlow or PyTorch library depending on the enviroment variables. In each library, Four GAN variants can be used to generate images: Pix2Pix, CycleGan, Pix2Pix with resnet and CycleGan with U-net.



### Example CycleGAN results
<img src="https://github.com/Falien164/kidneyGAN/blob/main/images/cyclegan_result.png" width="600" height="200">

### Extract images from slides

To extract images from slides you need .xml file with annotation. For this project ASAP was used to annotate all ROIs and dots where glomeruli is located. Use slider.py and all necessary argumets. Use --help to get information about arguments.

If you find this code useful in your research, please consider citing:
```
@inproceedings{lysik2022he,
  title={HE-to-PAS histological stain conversion by GAN in renal pathology},
  author={Lysik, Mateusz and Swiderska-Chadaj, Zaneta and Markiewicz, Tomasz and Les, Tomasz and Cierniak, Szczepan and Lorent, Malgorzata},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--7},
  year={2022},
  organization={IEEE}
}

@phdthesis{lysik2022gan,
  title={GAN application in the histological stain conversion},
  author={{\L}ysik, Mateusz},
  year={2022},
  school={Instytut Elektrotechniki Teoretycznej i System{\'o}w Informacyjno-Pomiarowych}
}
```
