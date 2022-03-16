
## Introduction
This code is a pose estimation method based on pixel-wise voting strategy. Our proposed method is improved based on PVNet. So we provide PVNet here for reference(https://github.com/zju3dv/clean-pvnet), and thanks again to Peng et al. for their excellent work.

We made the following change：  
- [x]  A DDL loss for learning unit vector-field is proposed for PVNet weak constraints


## Training and Testing
The code for installation, network training, and testing can be found in the PVNet link in the introduction section. The training and testing commands are shown below:<br>

### Take the benchvise as an example
1. Prepare the data related to `benchvise`:<br>
    
    >python run.py --type linemod cls_type benchvise
    
2. train:<br>

    >python train_net.py --cfg_file configs/linemod.yaml model yourmodel_dir cls_type benchvise
    
      
3. test with the uncertainty-driven PnP on Linemod and OCC-Linemod datasets:<br>

    >export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/csrc/uncertainty_pnp/lib<br>
    python run.py --type evaluate --cfg_file configs/linemod.yaml model yourmodel_dir cls_type benchvise test.un_pnp True<br>
    python run.py --type evaluate --cfg_file configs/linemod.yaml test.dataset LinemodOccTest model yourmodel_dir cls_type benchvise test.un_pnp True
    
   
## Schematic
 ![PVNet](https://github.com/YC0315/better_pvn/blob/f68a678f910756b554502a29853bd0ea20306c0b/views/PVNet.png)![PVNet_imp](https://github.com/YC0315/better_pvn/blob/f68a678f910756b554502a29853bd0ea20306c0b/views/PVNet_imp.png) 



