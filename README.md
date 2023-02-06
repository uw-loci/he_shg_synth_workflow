# Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis
Second-harmonic Generation Collagen Image Synthesis from Hematoxylin and Eosin Image Using Image-to-image Translation Neural Network  
Program for a complete H&amp;E-SHG synthesizing workflow  
Paper accepted at https://www.nature.com/articles/s42003-020-01151-5  
Intensity-based registration algorithm repository: https://github.com/uw-loci/shg_he_registration

|Input H&amp;E| Synthesized Collagen Image (SHG) |
|----------|--------|
|<img src="https://github.com/uw-loci/he_shg_synth_workflow/blob/master/thumbnails/he.jpg" width="320">|<img src="https://github.com/uw-loci/he_shg_synth_workflow/blob/master/thumbnails/shg.jpg" width="320">|

## download the [source code](https://github.com/uw-loci/he_shg_synth_workflow/archive/refs/heads/master.zip) and upzip it to a local drive,eg in windows
```
E:\he_shg_synth_workflow
```

## Required packages
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)  
then launch "Anaconda prompt", yielding eg in windows the base environment of conda as below:

(base) C:\Users\AccountName>

Required packages
```
Make the source code directory(eg E:\he_shg_synth_workflow) as the current working folder, using the following commands (in Windows)
cd E:\he_shg_synth_workflow
E:
Installation option 1: 
These two commands will change the current directory to:
(base) E:\he_shg_synth_workflow>
then install [Mambaforge windows 64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe)
follow the guide of the repo to install the packages:
conda env create --name synMF --file env.yml
conda activate synMF
pip3 install jpype1==1.3.0 (additional step)
pip3 install PyQt5 (additional step)
conda install -c jasonb857 importlib-metadata (additional step, version 3.10.1)
python main.py --pilot=0

Installation option 2:
first to install a PyImageJ environment with all packages satisfied
conda install mamba -n base -c conda-forge
mamba create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
Then, inside this environment, install dependencies for [PyTorch](https://pytorch.org/) and a bunch of other image processing packages such as sciki-image.

```

## Download example testing data, trained model weights, FIJI
Execute download.py
```  
python download.py
```
  
## Run demo
Execute main.py
```  
python main.py --pilot=0
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
