import os
import numpy as np
import cv2
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.metrics import mean_squared_error
from scipy.io import savemat
from sklearn.preprocessing import minmax_scale

# 导入自定义的评估函数
from evaluation_metrics import StructureMeasure, original_WFb, Fmeasure_calu, Enhancedmeasure

# ---- 1. Camouflage Map Path Setting ----
CamMapPath = '../Result/2020-CVPR-SINet/'  # Put model results in this folder.
Models = ['2020-CVPR-SINet']  # You can add other model like this format: Models = ['2019-ICCV-EGNet','2019-CVPR-CPD']
modelNum = len(Models)

# ---- 2. Ground-truth Datasets Setting ----
DataPath = '../Dataset/TestDataset/'
Datasets = ['CHAMELEON', 'CAMO', 'CPD1K', 'COD10K']  # You may also need other datasets, such as Datasets = ['CAMO','CPD1K']

# ---- 3. Results Save Path Setting ----
ResDir = './EvaluationResults/Result-CamObjDet/'
ResName = '_result.txt'  # You can change the result name.

Thresholds = np.linspace(1, 0, 256)
datasetNum = len(Datasets)

for d in range(datasetNum):
    dataset = Datasets[d]
    print(f'- Processing {d + 1}/{datasetNum}: {dataset} Dataset')
    
    ResPath = os.path.join(ResDir, f'{dataset}-mat/')
    os.makedirs(ResPath, exist_ok=True)
    resTxt = os.path.join(ResDir, f'{dataset}{ResName}')
    
    with open(resTxt, 'w') as fileID:
        for m in range(modelNum):
            model = Models[m]
            gtPath = os.path.join(DataPath, dataset, 'GT/')
            camPath = os.path.join(CamMapPath, model, dataset, '/')
            
            imgFiles = [f for f in os.listdir(camPath) if f.endswith('.png')]
            imgNUM = len(imgFiles)
            
            threshold_Fmeasure = np.zeros((imgNUM, len(Thresholds)))
            threshold_Emeasure = np.zeros((imgNUM, len(Thresholds)))
            threshold_Precion = np.zeros((imgNUM, len(Thresholds)))
            threshold_Recall = np.zeros((imgNUM, len(Thresholds)))
            Smeasure = np.zeros(imgNUM)
            wFmeasure = np.zeros(imgNUM)
            adpFmeasure = np.zeros(imgNUM)
            adpEmeasure = np.zeros(imgNUM)
            MAE = np.zeros(imgNUM)
            
            for i in range(imgNUM):
                print(f'- - Evaluating({dataset} Dataset,{model} Model): {i + 1}/{imgNUM}')
                name = imgFiles[i]
                
                # load gt
                gt = io.imread(os.path.join(gtPath, name))
                if gt.ndim > 2:
                    gt = rgb2gray(gt)
                gt = gt > 128
                
                # load camouflaged prediction
                cam = io.imread(os.path.join(camPath, name))
                
                # check size
                if cam.shape[:2] != gt.shape:
                    cam = resize(cam, gt.shape, anti_aliasing=True)
                    io.imsave(os.path.join(camPath, name), img_as_float(cam))
                    print(f'[Size Mismatching] Error occurs in the path: {os.path.join(camPath, name)}!!!')
                
                cam = img_as_float(cam)
                
                # normalize CamMap to [0, 1]
                cam = minmax_scale(cam.ravel(), feature_range=(0, 1)).reshape(cam.shape)
                
                # S-measure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
                Smeasure[i] = StructureMeasure(cam, gt)
                # Weighted F-measure metric published in CVPR'14 (How to evaluate the foreground maps?)
                wFmeasure[i] = original_WFb(cam, gt)
                
                # Using the 2 times of average of cam map as the threshold.
                threshold = 2 * cam.mean()
                _, _, adpFmeasure[i] = Fmeasure_calu(cam, gt, gt.shape, threshold)
                
                Bi_cam = np.zeros_like(cam)
                Bi_cam[cam >= threshold] = 1
                adpEmeasure[i] = Enhancedmeasure(Bi_cam, gt)
                
                threshold_F = np.zeros(len(Thresholds))
                threshold_E = np.zeros(len(Thresholds))
                threshold_Pr = np.zeros(len(Thresholds))
                threshold_Rec = np.zeros(len(Thresholds))
                for t in range(len(Thresholds)):
                    threshold = Thresholds[t]
                    threshold_Pr[t], threshold_Rec[t], threshold_F[t] = Fmeasure_calu(cam, gt, gt.shape, threshold)
                    
                    Bi_cam = np.zeros_like(cam)
                    Bi_cam[cam >= threshold] = 1
                    threshold_E[t] = Enhancedmeasure(Bi_cam, gt)
                
                threshold_Fmeasure[i, :] = threshold_F
                threshold_Emeasure[i, :] = threshold_E
                threshold_Precion[i, :] = threshold_Pr
                threshold_Recall[i, :] = threshold_Rec
                
                MAE[i] = mean_squared_error(gt, cam)
            
            column_F = threshold_Fmeasure.mean(axis=0)
            meanFm = column_F.mean()
            maxFm = column_F.max()
            
            column_Pr = threshold_Precion.mean(axis=0)
            column_Rec = threshold_Recall.mean(axis=0)
            
            column_E = threshold_Emeasure.mean(axis=0)
            meanEm = column_E.mean()
            maxEm = column_E.max()
            
            Sm = Smeasure.mean()
            wFm = wFmeasure.mean()
            
            adpFm = adpFmeasure.mean()
            adpEm = adpEmeasure.mean()
            mae = MAE.mean()
            
            savemat(os.path.join(ResPath, model), {
                'Sm': Sm, 'wFm': wFm, 'mae': mae, 'column_Pr': column_Pr, 'column_Rec': column_Rec,
                'column_F': column_F, 'adpFm': adpFm, 'meanFm': meanFm, 'maxFm': maxFm,
                'column_E': column_E, 'adpEm': adpEm, 'meanEm': meanEm, 'maxEm': maxEm
            })
            fileID.write(f'(Dataset:{dataset}; Model:{model}) Smeasure:{Sm:.3f}; wFmeasure:{wFm:.3f}; MAE:{mae:.3f}; '
                         f'adpEm:{adpEm:.3f}; meanEm:{meanEm:.3f}; maxEm:{maxEm:.3f}; adpFm:{adpFm:.3f}; '
                         f'meanFm:{meanFm:.3f}; maxFm:{maxFm:.3f}.\n')
            print(f'(Dataset:{dataset}; Model:{model}) Smeasure:{Sm:.3f}; wFmeasure:{wFm:.3f}; MAE:{mae:.3f}; '
                  f'adpEm:{adpEm:.3f}; meanEm:{meanEm:.3f}; maxEm:{maxEm:.3f}; adpFm:{adpFm:.3f}; '
                  f'meanFm:{meanFm:.3f}; maxFm:{maxFm:.3f}.\n')