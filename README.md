# AtomSegNet
AtomSegNet by DeepEM Lab@UCI

Please reference paper:  Scientific Reports, 11, 5386 (2021)  https://doi.org/10.1038/s41598-021-84499-w 

pytorch == 1.3.0
torchvision==0.4.0
scikit-image=0.18.3 (up to 0.18.3. Incompatible with 0.19 because watershed has been moved to a different module)


Try the following if you have trouble loading the app

conda install scikit-image=0.18.3

pip install scikit-image==0.18.3

### Running the App
```
python Code_Atom_Seg_Ui.py
```
In DOS terminal, enter
```
run
```

### Benchmark of the precision of AtomSegNe's atom localizer

We beat the golden standard--2d Gaussian fit
<p align="left"><img src="test_img/tem13.png" width="600"\></p>


### Examples

<p align="left"><img src="test_img/03_afterimage_8nm_crp_four_panel_guassianMask.png" width="600"\></p>

See more examples at Reports, 11, 5386 (2021)  https://doi.org/10.1038/s41598-021-84499-w 
