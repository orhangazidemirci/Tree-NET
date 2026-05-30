import torch
import torch.nn.functional as F
import time
import argparse
import csv
import os
import numpy as np
from datetime import datetime
from lib.models import call_model
from utils.dataloader import get_loader, test_dataset


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def warmup_real(model, image_root, gt_root, opt, pvt=False, n=10):
    """Warm up on real images so GPU clocks are stable before timing."""
    loader = test_dataset(image_root, gt_root, opt.trainsize, opt)
    num = min(n, len(os.listdir(gt_root)))
    with torch.no_grad():
        for _ in range(num):
            image, gt, _ = loader.load_data()
            image = image.cuda()
            if pvt:
                res, res1 = model(image)
            else:
                _ = model(image)
    torch.cuda.synchronize()



def load_test(path, opt):
    
    image_root = path + 'images/'
    gt_root    = path + 'masks/'
    
 
    loader = get_loader(image_root, gt_root,
                        batchsize=opt.batchsize,
                        trainsize=opt.trainsize,
                        args=opt,
                        augmentation=False)
    return loader
def run_inference(model, path, opt, batch_size):
    pvt= (opt.arch == 'pvt')
    opt.batchsize=batch_size
    loader=load_test(path, opt)
    model.eval()
    all_dice     = []
    all_latencies = []  # ms per batch
 
    with torch.no_grad():
        for i, pack in enumerate(loader, start=1):

            images, gts = pack
            images = images.cuda()
            gts    = gts.cuda()


            torch.cuda.synchronize()
            t0 = time.perf_counter()
 
            if pvt:
                res, res1 = model(images)
                res = F.upsample(res + res1, size=gts.shape[2:], mode='bilinear', align_corners=False)
            else:
                res = model(images)
                res = F.upsample(res, size=gts.shape[2:], mode='bilinear', align_corners=False)
 
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000)  # ms for this batch
 
            # res = res.sigmoid()
 
            # for i in range(res.shape[0]):
            #     pred   = res[i].cpu().numpy().squeeze()
            #     target = gts[i].cpu().numpy().squeeze()
            #     pred   = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            #     target = target / (target.max() + 1e-8)
 
            #     smooth       = 1
            #     intersection = (pred * target).sum()
            #     dice         = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
            #     all_dice.append(dice)
                # print(f'  [{len(all_dice)}]  dice: {dice:.4f}')
 
    # mean_dice      = float(np.mean(all_dice))
    lat_per_img_ms = float(np.mean(all_latencies)) / batch_size   # ms per image
    throughput     = round(1000.0 / lat_per_img_ms, 2)            # img/s
    lat_per_img_ms = round(lat_per_img_ms, 2)
 
    print(f'\nbatch_size={batch_size} | Treenet={opt.treenet}'
          f'Latency: {lat_per_img_ms} ms/img | Throughput: {throughput} img/s')
 
    return  lat_per_img_ms, throughput


# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(opt, dataset_paths):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results   = []
 
    experiments = {
        'ARCHS':         ['pvt', 'UNET', 'NestedUNET'],
        'BATCH_SIZES':   [1, 8],
        'DATASETS':      ['ISAC2018', 'CVC-ClinicDB'],
        'TREENET_MODES': [True, False],
    }
 
    for dataset in experiments['DATASETS']:
        opt.dataset = dataset
        path       = dataset_paths[dataset]
        image_root = path + 'images/'
        gt_root    = path + 'masks/'
 
        if not os.path.exists(gt_root):
            print(f'[SKIP] {dataset} — path not found: {gt_root}')
            continue
 
        for arch in experiments['ARCHS']:
            opt.arch = arch
            pvt      = (arch == 'pvt')
            if dataset== 'ISAC2018':
                o=1
            else:
                for treenet_mode in experiments['TREENET_MODES']:
                    opt.treenet = treenet_mode
                    if pvt:
                        opt.decoder_component = 'Decoder_x16'
                    else:
                        opt.decoder_component = 'Decoder_x4'

                    opt.encoder_path = f'./model_pth/{opt.encoder_component.lower()}_{dataset}.pt'
                    opt.decoder_path = f'./model_pth/{opt.decoder_component.lower()}_{dataset}.pt'
    
                    print(f'\n{"="*60}')
                    print(f'  arch={arch}  dataset={dataset}  treenet={treenet_mode}')
                    print(f'{"="*60}')
    
                    try:
                        model = call_model(opt).cuda()
                        model.eval()
                        
                        params = count_params(model)
                        print(f'  Params: {params:.2f}M')
                    except Exception as e:
                        print(f'  [SKIP] model load failed: {e}')
                        continue
    
                    warmup_real(model, image_root, gt_root, opt, pvt=pvt)
    
                    for bs in experiments['BATCH_SIZES']:
                        print(f'\n  -- batch_size={bs} --')
                        try:
                            lat_ms, throughput = run_inference(model, path, opt, batch_size=bs)
                            results.append({
                                'timestamp':    timestamp,
                                'arch':         arch,
                                'dataset':      dataset,
                                'treenet':      treenet_mode,
                                'batch_size':   bs,
                                'trainsize':    opt.trainsize,
                                'params_M':     round(params, 2),
                                'latency_ms':   lat_ms,
                                'throughput':   throughput,
                            })
                        except RuntimeError as e:
                            print(f'  [OOM or error] bs={bs}: {e}')
 
    # save CSV
    out_csv = f'./inference_results_{timestamp}.csv'
    if results:
        keys = list(results[0].keys())
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f'\nResults saved → {out_csv}')
 
    # summary
    print('\n' + '='*90)
    print(f'{"ARCH":<14} {"DATASET":<14} {"TREENET":<10} {"BS":<5} '
          f' {"LAT(ms)":<12} {"THRPUT(img/s)":<14}')
    print('-'*90)
    for row in results:
        print(f'{row["arch"]:<14} {row["dataset"]:<14} {str(row["treenet"]):<10} '
              f'{row["latency_ms"]:<12} {row["throughput"]:<14}')
 
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainsize',    type=int,   default=384)
    parser.add_argument('--clip',         type=float, default=0.5)
    parser.add_argument('--augmentation', default=False)
    opt = parser.parse_args()

    run_benchmark(opt)