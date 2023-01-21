# kidneyGAN
science article for university project (IEEE)
### ***WORK IN PROGRESS***
This project is about staining kidneys images (HE and PAS) and generating artificial images from one stain into another.

There will be implementation of Pix2Pix and CycleGAN methods on kidneys glomeruli. It will be possible to pick which learning should be used (supervised or unsupersived)


### Example CycleGAN results
<img src="https://github.com/Falien164/kidneyGAN/blob/main/images/cyclegan_result.png" width="600" height="200">

### Extract images from slides

To extract images from slides you need .xml file with annotation. For this project ASAP was used to annotate all ROIs and dots where glomeruli is located. Use slider.py and all necessary argumets. Use --help to get information about arguments.


## References
Łysik, Mateusz. GAN application in the histological stain conversion. Diss. Instytut Elektrotechniki Teoretycznej i Systemów Informacyjno-Pomiarowych, 2022.
