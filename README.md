# Brain-segmentation

## Introduction
The central nervous system (CNS) is the part of the nervous system consisting of the brain and spinal cord. It integrates information it receives from and coordinates and influences the activity of all parts of the body. The brain is contained in, and protected by, the skull bones of the head. The cerebrum is the largest part of the human brain. It is divided into two cerebral hemispheres. The cerebral cortex is an outer layer of grey matter, covering the core of white matter.

White matter is composed of bundles, which connect various gray matter areas (the locations of nerve cell bodies) of the brain to each other and carry nerve impulses between neurons. Myelin acts as an insulator, which allows electrical signals to jump, rather than coursing through the axon, increasing the speed of transmission of all nerve signals. The other main component of the brain is grey matter (actually pinkish tan due to blood capillaries), which is composed of neurons. Cerebrospinal fluid is a colorless transcellular fluid that circulates the brain in the subarachnoid space, in the ventricular system, and in the central canal of the spinal cord.

Segmentation of brain tissues in MRI image has a number of applications in diagnosis, surgical planning, and treatment of brain abnormalities. However, it is a time-consuming task to be performed by medical experts. In addition to that, it is challenging due to intensity overlap between the different tissues caused by the intensity homogeneity and artifacts inherent to MRI. To minimize this effect, it was proposed to apply histogram based preprocessing. The goal of this project is to develop a robust and automatic segmentation of White Matter (WM) and Gray Matter (GM)) and Cerebrospinal Fluid (CSF) of the human brain. 

To tackle the problem, we have proposed  Convolutional Neural Network (CNN) based approach and probabilistic Atlas. U-net is one of the most commonly used and best-performing architecture in medical image segmentation, and we have used both 2D and 3D versions.   The performance was evaluated using DSC, HD and AVD.

## Dataset
For this project, the proposed solutions will be evaluated on the well-known IBSR18 dataset which is one of the
standard datasets for tissue quantification and segmentation evaluation. The dataset consists of 18 MRI volumes including: ten volumes for training (red), five for validation (blue) and three for testing (yellow). For the training and validation images, the corresponding ground truth (GT) is provided, while for the testing set it will not be available.

By the end of this project, the results in this testing were submitted to perform a competition with all the other groups. The
ranking will take into account the performance of the algorithm based on Dice (DSC), Hausdorff distance (HD) and average volumetric difference (AVD), as commonly defined in other challenges (e.g. MICCAI2012, MRBrainS13, and iSeg2017). 
