
## ����
- PyTorch>1.10  
- OpenCV
- Matplotlib 3.3.4 
- opencv-python 
- pyyaml
- tqdm
- numpy
- torchvision

## ���ݼ�

- OmniSR_X4_DIV2K.zip ��[baidu cloud](https://pan.baidu.com/s/1kGasS_wslZy4OyzaHTukvg) (passwd: sjtu) , 
[Google driver](https://drive.google.com/file/d/1VoPUw0SRnCPAU8_R5Ue15bn2gwSBr97g/view?usp=sharing)

- OmniSR_X4_DF2K.zip ��[baidu cloud](https://pan.baidu.com/s/1ovxRa4-wOKZLq_nO6hddsg) (passwd: sjtu) , 
[Google driver](https://drive.google.com/file/d/17rJXJHBYt4Su8cMDMh-NOWMBdE6ki5em/view?usp=sharing)|

-  benchmark  [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), ���Ƶ� ```./benchmark/```. 
 

## ѵ��

### ʾ��: ѵ��ģ��DIV2K@X4:

- Step1, �������ݼ� [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (```Train Data Track 1 bicubic downscaling x4 (LR images)``` and ```Train Data (HR images)```), Ȼ���������ļ����������ݼ�·�� ```./env/env.json: Line 8: "DIV2K":"TO YOUR DIV2K ROOT PATH"```

- Step2, ���� benchmark ([baidu cloud](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), ���Ƶ� ```./benchmark/```. 

- Step3, ʹ������pythonָ�ʼѵ��:
```
python train.py -v "OmniSR_X4_DIV2K" -p train --train_yaml "train_OmniSR_X4_DIV2K.yaml"
```

## ����

### ʾ��: ����ѵ����ģ��DF2K@X4:
- ����ѡ������Ԥѵ��ģ�͵�Ŀ¼�� ```./train_logs/```:
- Step 1, ʹ������pythonָ��������ɵ�ͼ��������� ```./SR```

```
python test.py -v "OmniSR_X4_DF2K" -s 994 -t tester_Matlab --test_dataset_name "Urban100"
```

- Step2, ִ�и�Ŀ¼�е�`Evaluate_PNR_SSIM.m`�ű����Ի�ȡ�����б���Ľ�������޸�```Line 8 (Evaluate_PSNR_SSIM.m): methods = {'OmniSR_X4_DF2K'};``` and ```Line 10 (Evaluate_PSNR_SSIM.m): dataset = {'Urban100'};```��ƥ������������ģ��/���ݼ����ơ�