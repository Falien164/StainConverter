# StainConverter
science article for university project (IEEE)
### ***WORK IN PROGRESS***
This project is about staining kidneys images (HE and PAS) and generating artificial images from one stain into another.


### Example CycleGAN results
<img src="https://github.com/Falien164/kidneyGAN/blob/main/images/cyclegan_result.png" width="600" height="200">

### Extract images from slides

To extract images from slides you need .xml file with annotation. For this project ASAP was used to annotate all ROIs and dots where glomeruli is located. Use slider.py and all necessary argumets. Use --help to get information about arguments.

If you find this code useful in your research, please consider citing:
```
@phdthesis{lysik2022gan,
  title={GAN application in the histological stain conversion},
  author={{\L}ysik, Mateusz},
  year={2022},
  school={Instytut Elektrotechniki Teoretycznej i System{\'o}w Informacyjno-Pomiarowych}
}
```
