# FreeCus

This repository is the official implementation of [FreeCus](#) [ICCV-2025].

> **FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers** <br>
> Yanbing Zhang, Zhe Wang, Qin Zhou, Mengping Yang<br>
> [pdf](#)

<div>
<p align="center">
<img src='assets/teaser.jpg' align="center" width=900>
</p>
</div>

## Installation
We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```
# Create a new conda environment
conda create -n freecus python=3.10
conda activate freecus

# Install other dependencies
pip install -r requirements.txt
```

## Download
You can ​download models directly from Hugging Face. This includes [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), and [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet).

## Quickstart
```
python3 infer.py --qwen2_vl_path {The path of model Qwen2-VL} \
                 --qwen2_5_path {The path of model Qwen2.5} \
                 --birefnet_path {The path of model BiRefNet} \
                 --flux_path {The path of model Flux.1-dev} \
                 --input_image 'assets/stork.jpg' \
                 --prompt "A stork with glowing wings dancing on the waves of the northern lights" \
                 --subject_word 'stork' 
```

## Contact Us
**Yanbing Zhang**: [zhangyanbing@mail.ecust.edu.cn](mailto:zhangyanbing@mail.ecust.edu.cn)  
**Mengping Yang**: [kobeshegu@gmail.com](mailto:kobeshegu@gmail.com)  

## BibTeX
```
@article{zhang2025freecus,
  title={FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers},
  author={Zhang, Yanbing and Wang, Zhe and Zhou, Qin and Yang, Mengping},
  journal={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2025}
}
```

## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 
