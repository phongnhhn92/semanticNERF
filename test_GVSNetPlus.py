from opt import get_opts
from train_GVSNETPlus import NeRFSystem
from utils import load_ckpt
import torch,tqdm,os
from datasets import dataset_dict
from datasets.carla_utils.utils import SaveSemantics
from datasets.ray_utils import getRandomRays
from torch.utils.data import DataLoader
from utils.visualization import save_depth
from torchvision.utils import save_image

if __name__ == '__main__':
    hparams = get_opts()
    checkpoint_path = './GVSPlus/ckpts/exp_GVSPlus_AlphaSampler_withSkipConnection/epoch=20-val_loss=0.00.ckpt'
    model = NeRFSystem(hparams)
    model_dict = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)
    model.cuda().eval()
    
    dataset = dataset_dict[hparams.dataset_name]
    test_dataset = dataset(hparams, split='test')
    
    test_dataloader = DataLoader(test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)
    print('Datasize size: {}'.format(len(test_dataloader)))

    
    # Test result folder
    test_folder_output = './output/'
    if os.path.exists(test_folder_output) is not True:
        os.mkdir(test_folder_output)

    with torch.no_grad():
        for i,data in tqdm(enumerate(test_dataloader)):
            # print('Processing {}/{}'.format(i, len(test_dataloader)))
            # move to gpu
            input={}
            for k,v in data.items():
                input[k] = v.cuda()
            
            #Inference
            results = model(input,training = False)   

            save_semantic = SaveSemantics('carla')
            W, H = hparams.img_wh
            input_img = data['input_img'][0].cpu()
            input_img = input_img * 0.5 + 0.5

            input_seg = torch.argmax(data['input_seg'][0], dim=0).cpu()
            input_seg = torch.from_numpy(save_semantic.to_color(input_seg)).permute(2, 0, 1)
            input_seg = input_seg / 255.0           
            

            target_img = data['target_img'][0].cpu()
            target_img = target_img * 0.5 + 0.5

            target_seg = torch.argmax(data['target_seg'][0], dim=0).cpu()
            target_seg = torch.from_numpy(save_semantic.to_color(target_seg)).permute(2, 0, 1)
            target_seg = target_seg / 255.0

            stack = torch.stack([input_img, input_seg, target_img, target_seg])
            save_image(stack, test_folder_output+'{}_gt.png'.format(i))

            pred_seg = torch.argmax(results['semantic_nv'].squeeze(), dim=0).cpu()
            pred_seg = torch.from_numpy(save_semantic.to_color(pred_seg)).permute(2, 0, 1)
            pred_seg = pred_seg / 255.0

            pred_disp = save_depth(results['disp_nv'].squeeze().cpu())
            baseline = hparams.stereo_baseline
            fx = 128.0
            pred_depth_cvt = baseline * fx / results['depth']
            pred_depth = save_depth(pred_depth_cvt.squeeze().view(H, W).cpu())
            pred_rgb = results['rgb'].squeeze().permute(1, 0).view(3, H, W).cpu()

            stack_pred = torch.stack([pred_rgb, pred_seg, pred_disp, pred_depth])
            save_image(stack_pred, test_folder_output+'{}_pred.png'.format(i))
    print('Done')

