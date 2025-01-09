
## 依赖
- PyTorch>1.10  
- OpenCV
- Matplotlib 3.3.4 
- opencv-python 
- pyyaml
- tqdm
- numpy
- torchvision

## 数据集

- OmniSR_X4_DIV2K.zip ：[baidu cloud](https://pan.baidu.com/s/1kGasS_wslZy4OyzaHTukvg) (passwd: sjtu) , 
[Google driver](https://drive.google.com/file/d/1VoPUw0SRnCPAU8_R5Ue15bn2gwSBr97g/view?usp=sharing)

- OmniSR_X4_DF2K.zip ：[baidu cloud](https://pan.baidu.com/s/1ovxRa4-wOKZLq_nO6hddsg) (passwd: sjtu) , 
[Google driver](https://drive.google.com/file/d/17rJXJHBYt4Su8cMDMh-NOWMBdE6ki5em/view?usp=sharing)|

-  benchmark  [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), 复制到 ```./benchmark/```. 
 

## 训练

### 示例: 训练模型DIV2K@X4:

- Step1, 下载数据集 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (```Train Data Track 1 bicubic downscaling x4 (LR images)``` and ```Train Data (HR images)```), 然后在配置文件中设置数据集路径 ```./env/env.json: Line 8: "DIV2K":"TO YOUR DIV2K ROOT PATH"```

- Step2, 下载 benchmark ([baidu cloud](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), 复制到 ```./benchmark/```. 

- Step3, 使用以下python指令开始训练:
```
python train.py -v "OmniSR_X4_DIV2K" -p train --train_yaml "train_OmniSR_X4_DIV2K.yaml"
```

## 评估

### 示例: 评估训练的模型DF2K@X4:
- （可选）下载预训练模型到目录下 ```./train_logs/```:
- Step 1, 使用以下python指令，并将生成的图像放置在中 ```./SR```

```
python test.py -v "OmniSR_X4_DF2K" -s 994 -t tester_Matlab --test_dataset_name "Urban100"
```

- Step2, 执行根目录中的`Evaluate_PNR_SSIM.m`脚本，以获取论文中报告的结果。请修改```Line 8 (Evaluate_PSNR_SSIM.m): methods = {'OmniSR_X4_DF2K'};``` and ```Line 10 (Evaluate_PSNR_SSIM.m): dataset = {'Urban100'};```以匹配上面评估的模型/数据集名称。