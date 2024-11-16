from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from skimage import io
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm


def fmeasure_calu(sMap, gtMap, gtsize, threshold):
    """
    计算 F-measure。

    :param sMap: 显著图（numpy 数组）
    :param gtMap: 地面真值图像（numpy 数组）
    :param gtsize: 地面真值图像的尺寸
    :param threshold: 阈值
    :return: Precision, Recall 和 F-measure
    """
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros(gtsize, dtype=bool)
    Label3[sMap >= threshold] = True

    NumRec = np.sum(Label3)
    LabelAnd = np.logical_and(Label3, gtMap)
    NumAnd = np.sum(LabelAnd)
    num_obj = np.sum(gtMap)

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
    else:
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        FmeasureF = (1.3 * PreFtem * RecallFtem) / (0.3 * PreFtem + RecallFtem)

    return PreFtem, RecallFtem, FmeasureF


def emeasure(FM, GT):
    """
    计算增强对齐度量（E-measure）。

    :param FM: 二值前景图（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 增强对齐得分
    """
    FM = FM.astype(bool)
    GT = GT.astype(bool)

    # 使用 double 进行计算
    dFM = FM.astype(np.float64)
    dGT = GT.astype(np.float64)

    # 特殊情况
    if np.sum(dGT) == 0:  # 如果 GT 完全是黑色
        enhanced_matrix = 1.0 - dFM  # 只计算黑色区域的交集
    elif np.sum(~GT) == 0:  # 如果 GT 完全是白色
        enhanced_matrix = dFM  # 只计算白色区域的交集
    else:
        # 正常情况
        # 1. 计算对齐矩阵
        align_matrix = alignment_term(dFM, dGT)
        # 2. 计算增强对齐矩阵
        enhanced_matrix = enhanced_alignment_term(align_matrix)

    # 3. 计算 E-measure 得分
    w, h = GT.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)

    return score


def alignment_term(dFM, dGT):
    """
    计算对齐矩阵

    :param dFM: double 类型的前景图
    :param dGT: double 类型的地面真值图
    :return: 对齐矩阵
    """
    # 计算全局均值
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)

    # 计算偏差矩阵
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT

    # 计算对齐矩阵
    align_matrix = (
        2
        * (align_GT * align_FM)
        / (align_GT * align_GT + align_FM * align_FM + np.finfo(float).eps)
    )

    return align_matrix


def enhanced_alignment_term(align_matrix):
    """
    计算增强对齐矩阵

    :param align_matrix: 对齐矩阵
    :return: 增强对齐矩阵
    """
    enhanced = ((align_matrix + 1) ** 2) / 4
    return enhanced


def cal_mae(smap, gt_img):
    """
    计算显著图和地面真值图像之间的平均绝对误差（MAE）。

    :param smap: 显著图（numpy 数组）
    :param gt_img: 地面真值图像（numpy 数组）
    :return: MAE 值
    """
    if smap.shape != gt_img.shape:
        raise ValueError("显著图和地面真值图像的尺寸不同！")

    if not np.issubdtype(gt_img.dtype, np.bool_):
        gt_img = gt_img > 128

    smap = smap.astype(np.float64)
    gt_img = gt_img.astype(np.float64)

    mae = mean_absolute_error(gt_img, smap)

    return mae


def original_wfb(FG, GT):
    """
    计算加权 F-beta 度量（Weighted F-beta measure）。

    :param FG: 二值/非二值前景图，值范围在 [0, 1] 之间（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 加权 F-beta 得分
    """
    if not isinstance(FG, np.ndarray) or FG.dtype != np.float64:
        raise ValueError("FG 应该是 double 类型的 numpy 数组")
    if np.max(FG) > 1 or np.min(FG) < 0:
        raise ValueError("FG 应该在 [0, 1] 范围内")
    if not isinstance(GT, np.ndarray) or GT.dtype != np.bool_:
        raise ValueError("GT 应该是逻辑类型的 numpy 数组")

    dGT = GT.astype(np.float64)  # 使用 double 进行计算

    E = np.abs(FG - dGT)

    Dst, IDXT = distance_transform_edt(dGT, return_indices=True)
    # 像素依赖性
    K = gaussian_filter(np.ones((7, 7)), 5)
    Et = E.copy()
    Et[~GT] = Et[tuple(IDXT[:, ~GT])]  # 正确处理前景区域边缘
    EA = gaussian_filter(Et, sigma=5)
    MIN_E_EA = E.copy()
    MIN_E_EA[GT & (EA < E)] = EA[GT & (EA < E)]
    # 像素重要性
    B = np.ones_like(GT, dtype=np.float64)
    B[~GT] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * Dst[~GT])
    Ew = MIN_E_EA * B

    TPw = np.sum(dGT) - np.sum(Ew[GT])
    FPw = np.sum(Ew[~GT])

    R = 1 - np.mean(Ew[GT])  # 加权召回率
    P = TPw / (np.finfo(float).eps + TPw + FPw)  # 加权精度

    Q = 2 * (R * P) / (np.finfo(float).eps + R + P)  # Beta=1

    return Q


def s_object(prediction, GT):
    """
    计算前景图和地面真值之间的对象相似度。

    :param prediction: 二值/非二值前景图，值范围在 [0, 1] 之间（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 对象相似度得分
    """
    prediction_fg = prediction.copy()
    prediction_fg[~GT] = 0
    O_FG = object_similarity(prediction_fg, GT)

    prediction_bg = 1.0 - prediction
    prediction_bg[GT] = 0
    O_BG = object_similarity(prediction_bg, ~GT)

    u = np.mean(GT)
    Q = u * O_FG + (1 - u) * O_BG

    return Q


def object_similarity(prediction, GT):
    """
    计算对象相似度。

    :param prediction: 前景图（numpy 数组）
    :param GT: 地面真值图（numpy 数组）
    :return: 对象相似度得分
    """
    if prediction.size == 0:
        return 0

    if np.issubdtype(prediction.dtype, np.integer):
        prediction = prediction.astype(np.float64)

    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        raise ValueError("prediction 应该是 double 类型的 numpy 数组")
    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError("prediction 应该在 [0, 1] 范围内")
    if not isinstance(GT, np.ndarray) or GT.dtype != np.bool_:
        raise ValueError("GT 应该是逻辑类型的 numpy 数组")

    x = np.mean(prediction[GT])
    sigma_x = np.std(prediction[GT])

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + np.finfo(float).eps)
    return score


def s_region(prediction, GT):
    """
    计算前景图和地面真值之间的区域相似度。

    :param prediction: 二值/非二值前景图，值范围在 [0, 1] 之间（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 区域相似度得分
    """
    X, Y = centroid(GT)

    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divide_gt(GT, X, Y)
    prediction_1, prediction_2, prediction_3, prediction_4 = divide_prediction(
        prediction, X, Y
    )

    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)

    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q


def centroid(GT):
    """
    计算地面真值的质心。

    :param GT: 二值地面真值图（numpy 数组）
    :return: 质心的坐标 (X, Y)
    """
    rows, cols = GT.shape

    if np.sum(GT) == 0:
        X = cols // 2
        Y = rows // 2
    else:
        total = np.sum(GT)
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1)
        X = int(np.round(np.sum(np.sum(GT, axis=0) * i) / total))
        Y = int(np.round(np.sum(np.sum(GT, axis=1) * j) / total))

    return X, Y


def divide_gt(GT, X, Y):
    """
    根据地面真值的质心将其分为 4 个区域，并返回权重。

    :param GT: 二值地面真值图（numpy 数组）
    :param X: 质心的 X 坐标
    :param Y: 质心的 Y 坐标
    :return: 4 个区域和权重
    """
    hei, wid = GT.shape
    area = wid * hei

    LT = GT[:Y, :X]
    RT = GT[:Y, X:]
    LB = GT[Y:, :X]
    RB = GT[Y:, X:]

    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei - Y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4


def divide_prediction(prediction, X, Y):
    """
    根据地面真值的质心将前景图分为 4 个区域。

    :param prediction: 二值/非二值前景图（numpy 数组）
    :param X: 质心的 X 坐标
    :param Y: 质心的 Y 坐标
    :return: 4 个区域
    """
    hei, wid = prediction.shape

    LT = prediction[:Y, :X]
    RT = prediction[:Y, X:]
    LB = prediction[Y:, :X]
    RB = prediction[Y:, X:]

    return LT, RT, LB, RB


def ssim(prediction, GT):
    """
    计算前景图和地面真值之间的结构相似度（SSIM）。

    :param prediction: 二值/非二值前景图，值范围在 [0, 1] 之间（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 结构相似度得分
    """
    dGT = GT.astype(np.float64)

    hei, wid = prediction.shape
    N = wid * hei

    x = np.mean(prediction)
    y = np.mean(dGT)

    sigma_x2 = np.sum((prediction - x) ** 2) / (N - 1 + np.finfo(float).eps)
    sigma_y2 = np.sum((dGT - y) ** 2) / (N - 1 + np.finfo(float).eps)

    sigma_xy = np.sum((prediction - x) * (dGT - y)) / (N - 1 + np.finfo(float).eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q


def structure_measure(prediction, GT):
    """
    计算前景图和地面真值之间的结构相似度。

    :param prediction: 二值/非二值前景图，值范围在 [0, 1] 之间（numpy 数组）
    :param GT: 二值地面真值图（numpy 数组）
    :return: 计算的相似度得分
    """
    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        raise ValueError("The prediction should be double type...")
    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError("The prediction should be in the range of [0 1]...")
    if not isinstance(GT, np.ndarray) or GT.dtype != np.bool_:
        raise ValueError("GT should be logical type...")

    y = np.mean(GT)

    if y == 0:  # if the GT is completely black
        x = np.mean(prediction)
        Q = 1.0 - x  # only calculate the area of intersection
    elif y == 1:  # if the GT is completely white
        x = np.mean(prediction)
        Q = x  # only calculate the area of intersection
    else:
        alpha = 0.5
        Q = alpha * s_object(prediction, GT) + (1 - alpha) * s_region(prediction, GT)
        if Q < 0:
            Q = 0

    return Q


def load_npy_files_from_folder(folder):
    npy_files = {}
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            npy_files[filename] = np.load(os.path.join(folder, filename))
    return npy_files


def load_and_convert_png_files_from_folder(folder, save_folder):
    png_files = {}
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = io.imread(os.path.join(folder, filename))
            npy_filename = filename.replace(".png", ".npy")
            np.save(os.path.join(save_folder, npy_filename), img)
            png_files[npy_filename] = img
    return png_files


def process_file(filename, smap_files, gt_files):
    if filename in gt_files:
        smap = smap_files[filename].astype(np.float64)
        gt_img = gt_files[filename].astype(np.bool_)

        mae = cal_mae(smap, gt_img)
        Emeasure = emeasure(smap, gt_img)
        PreFtem, RecallFtem, FmeasureF = fmeasure_calu(smap, gt_img, gt_img.shape, 0.5)
        F_beta = original_wfb(smap, gt_img)
        s_object_score = s_object(smap, gt_img)
        s_region_score = s_region(smap, gt_img)
        structure_measure_score = structure_measure(smap, gt_img)

        return (
            mae,
            Emeasure,
            FmeasureF,
            F_beta,
            s_object_score,
            s_region_score,
            structure_measure_score,
        )
    return None


if __name__ == "__main__":
    smap_folder = "Result/2020-CVPR-SINet-New/COD10K"  # 替换为显著图文件夹路径
    gt_folder = "COD10K/GT"  # 替换为地面真值图像文件夹路径
    gt_npy_folder = "COD10K/GT_NPY"  # 保存转换后的地面真值图像的文件夹路径

    smap_files = load_npy_files_from_folder(smap_folder)
    gt_files = load_npy_files_from_folder(gt_npy_folder)
    # gt_files = load_and_convert_png_files_from_folder(gt_folder, gt_npy_folder)

    mae_list = []
    emeasure_list = []
    fmeasure_list = []
    fbeta_list = []
    s_object_list = []
    s_region_list = []
    structure_measure_list = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda filename: process_file(filename, smap_files, gt_files),
                    smap_files,
                ),
                total=len(smap_files),
            )
        )

    for result in results:
        if result:
            (
                mae,
                Emeasure,
                FmeasureF,
                F_beta,
                s_object_score,
                s_region_score,
                structure_measure_score,
            ) = result
            mae_list.append(mae)
            emeasure_list.append(Emeasure)
            fmeasure_list.append(FmeasureF)
            fbeta_list.append(F_beta)
            s_object_list.append(s_object_score)
            s_region_list.append(s_region_score)
            structure_measure_list.append(structure_measure_score)

    # 计算平均值
    avg_mae = np.mean(mae_list)
    avg_emeasure = np.mean(emeasure_list)
    avg_fmeasure = np.mean(fmeasure_list)
    avg_fbeta = np.mean(fbeta_list)
    avg_s_object = np.mean(s_object_list)
    avg_s_region = np.mean(s_region_list)
    avg_structure_measure = np.mean(structure_measure_list)

    # 输出平均值结果
    result = (
        f"Average MAE: {avg_mae:.4f}, Average E-measure: {avg_emeasure:.4f}, "
        f"Average F-measure: {avg_fmeasure:.4f}, Average F_beta: {avg_fbeta:.4f}, "
        f"Average S_object: {avg_s_object:.4f}, Average S_region: {avg_s_region:.4f}, "
        f"Average Structure_measure: {avg_structure_measure:.4f}"
    )
    print(result)
    
