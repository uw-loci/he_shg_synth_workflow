# Second-harmonic Generation Collagen Image Synthesis from Hematoxylin and Eosin Image Using Image-to-image Translation Neural Network
Program for a complete H&amp;E-SHG synthesizing workflow  
Paper accepted at https://www.nature.com/articles/s42003-020-01151-5

Intensity-based registration algorithm repository: https://github.com/uw-loci/shg_he_registration

|Input H&amp;E| Synthesized Collagen Image (SHG) |
|----------|--------|
|<img src="https://github.com/uw-loci/he_shg_synth_workflow/blob/master/thumbnails/he.jpg" width="320">|<img src="https://github.com/uw-loci/he_shg_synth_workflow/blob/master/thumbnails/shg.jpg" width="320">|

## Required packages
Install required packages in a virtual environment, commands for anaconda/miniconda (https://www.anaconda.com/distribution/) are listed
* python==3.6.x
```
  conda create --name [NAME_ENV] python=3.6
  conda activate [NAME_ENV]
```
* matplotlib==3.1.2 
```
  conda install -c conda-forge matplotlib=3.1.2
```  
* numpy==1.17.4
```
  conda install -c anaconda numpy=1.17.4
```  
* pandas==0.25.3
```
  conda install -c anaconda pandas=0.25.3
``` 
* Pillow==5.3.0
```
  conda install -c anaconda pillow=5.3.0
```  
* pyimagej==0.4.0
```
conda install -c conda-forge pyimagej
```
* scikit-image==0.16.2
```
  conda install -c anaconda scikit-image=0.16.2
```  
* tqdm==4.42.0
```
  conda install -c conda-forge tqdm=4.42.0
```  
* pytorch>=1.3.1
```
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```  
## Download example testing data, trained model weights, FIJI
Execute download.py
```  
python download.py
```
  
## Run demo
Execute main.py
```  
python main.py
```

Output images are saved in "output_test_default" folder by default.
### Argumenets for main.py
```
[--use-cuda]          # 1: use GPU, 0: use CPU                            default: (int) 1
[--which-gpu]         # index of the GPU                                  default: (int) 0
[--input-folder]      # name of input folder (input_test_[NAME])          default: (str) default
[--intensity]         # output intensity rescale                          default: (tuple) (20, 180)
[--pilot]             # 1: process the first image, 0: process all images default: (int) 0
```
Test customized images:

1. Create a folder named "input_test_[NAME]" containing input images (images from a 40x Aperio CS2 scanner are recommended).
2. Execute main.py with option "--input-folder=[NAME]".
```
python main.py --input-folder=[NAME]
```
3. Output images are saved in "output_test_[NAME]" folder.
  
## Citations
```
@article{keikhosravi_non-disruptive_2020,
	title = {Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis},
	volume = {3},
	copyright = {2020 The Author(s)},
	issn = {2399-3642},
  url = {https://www.nature.com/articles/s42003-020-01151-5},
	doi = {10.1038/s42003-020-01151-5},
	number = {1},
	journal = {Communications Biology},
	author = {Keikhosravi, Adib and Li, Bin and Liu, Yuming and Conklin, Matthew W. and Loeffler, Agnes G. and Eliceiri, Kevin W.},
	month = jul,
	year = {2020},
	note = {Publisher: Nature Publishing Group},
	pages = {1--12}
}
```
