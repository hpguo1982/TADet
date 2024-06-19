import os

import torch
from mmengine.runner import Runner
from mmseg.apis import init_model
from mmengine.config import Config
from PIL import Image


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def save_output(image_name, pred, d_dir, ori_size):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')

	imo = im.resize(ori_size, resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')


device='cuda:0'
#for loading test set
config_path_test = 'configs/_base_/datasets_test/sd_saliency_900_test.py'

#for loading model
config_path = './configs/msamff/sssd_tri_attention/fcn-d6_r50-d16_4xb2-80k_sd_saliency_900-256x256.py'
checkpoint_path = './pretrained/model.pth'

#for save results
pred_maps_dir = '../data/SD-saliency-900/evaluation/pred_maps_sel'
dataset = 'sp0.0'
prediction_dir = pred_maps_dir + '/model/' + dataset + '/'




def predict():

    if not os.path.exists(prediction_dir):
        # 如果不存在，创建目录
        os.makedirs(prediction_dir)
        print(f"目录 {prediction_dir} 已创建。")

    #load data set
    cfg = Config.fromfile(config_path_test)
    test_dataloader = cfg.get('test_dataloader')
    test_dataloader.dataset['_scope_'] = 'mmseg'
    dataLoader = Runner.build_dataloader(test_dataloader)

    #load trained model
    model = init_model(config_path, checkpoint_path, device=device)

    #save prediction
    for i, data_sample in enumerate(dataLoader):
        prediction = model.test_step(data_sample)
        pred = normPRED(prediction[0].seg_logits.data)
        save_output(prediction[0].img_path, pred, prediction_dir, prediction[0].ori_shape)


if __name__ == '__main__':
    predict()



