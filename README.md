# **Human Ergonomics Analysis with RULA and HumanPoseGAN**

## **Project Overview**
This repository is dedicated to my Master's thesis, which involves applying advanced computational methods in the field of human ergonomics. The project integrates **Rapid Upper Limb Assessment (RULA)**, a renowned ergonomic evaluation technique, with state-of-the-art machine learning models, specifically **HumanPoseGAN**, to assess and optimize human postures for ergonomic safety.
###  NOTE this project is still in development...
## **Methodology**
- **RULA (Rapid Upper Limb Assessment)**: Utilized as a foundational framework for ergonomic risk assessment. This method provides a systematic process for evaluating the risk of musculoskeletal disorders in the upper extremities.
- **Gradient-Based Optimization**: Employed to iteratively adjust and optimize human 3D poses, reducing ergonomic risk as per RULA guidelines.
- **HumanPoseGAN**: Integrated to enhance the precision of pose generation and refinement, leveraging the capabilities of Generative Adversarial Networks in understanding and replicating human postures.

## **Tools and Technologies**
- **PyTorch**: The primary deep learning framework used for model training and gradient computations.
- **Detectron2**: Applied for 2D pose estimation, providing the foundational data for further 3D pose processing.
- **VideoPose3D**: Utilized for converting 2D poses into 3D representations.
- **Custom Ergonomic Assessment Module**: Developed specifically for this project to apply RULA in a computational context.
- **Optimization Algorithms**: Various algorithms including Adam, SGD, and RMSprop are used for optimizing the 3D poses based on ergonomic criteria.

## **Project Structure**
- `Wrapper_2Dpose` and `Wrapper_3Dpose`: Modules for processing and converting video data into 2D and 3D pose representations.
- `Ergonomics_Torch`: A custom module that integrates ergonomic assessment into the pose optimization process.
- `HumanPoseDiscriminator`: Part of HumanPoseGAN, used for assessing and refining generated poses.

## **Contributions and Usage**
This project is part of ongoing research in human ergonomics and pose optimization. Contributions, suggestions, and discussions are welcome. For more detailed information on the methodologies and usage, refer to the specific documentation provided within each module.
