# M-SAM \[[Paper](https://arxiv.org/pdf/2403.05912)]

<a src="https://img.shields.io/badge/cs.CV-2403.05912-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/pdf/2403.05912"> 
<img src="https://img.shields.io/badge/cs.CV-2403.05912-b31b1b?logo=arxiv&logoColor=red">

We introduce Mask-Enhanced SAM (M-SAM), an innovative architecture tailored for 3D tumor lesion segmentation. This method is elaborated on the paper [Mask-Enhanced Segment Anything Model for Tumor Lesion Semantic Segmentation](https://arxiv.org/pdf/2403.05912).

## 🌟 Highlights
-  We introduce a novel Mask-Enhanced SAM (M-SAM) architecture to explore the application of [SAM](https://github.com/facebookresearch/segment-anything) in the medical domain, validating its effectiveness in tumor lesion segmentation.
-  We propose a Mask-Enhanced Adapter (MEA) to align the positional information of the prompt with the semantic information of the input image, optimizing precise guidance for mask prediction. Based on the design of the MEA, we further implement an iterative refining scheme to refine masks, yielding improved performances.
- With updates to only about 20% of the parameters, our model outperforms state-of-the-art medical image segmentation methods on five tumor lesion segmentation benchmarks. Additionally, we validate the effectiveness of our method in domain transferring.

## 👉 A Quick Overview
M-SAM consists of multiple stages of iterative refinement, which makes it possible to refine the predicted segmentation masks iteratively, thus obtaining more accurate segmentation boundaries progressively.
<p align="center"><img width="800" alt="image" src="https://github.com/nanase1025/M-SAM/blob/main/assets/architecture.jpg"></p> 
Our MEA is proposed to aggregate the image embedding with corresponding mask, so that the updated image embedding can perceive position priors of the lesion regions.
<p align="center"><img width="800" alt="image" src="https://github.com/nanase1025/M-SAM/blob/main/assets/MEA.jpg"></p> 

## 👉 Requirement
 Install the environment:
 ```bash
pip install -r requirements.txt
```
Then download [SAM checkpoint](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view), and put it at .work_dir/SAM/
