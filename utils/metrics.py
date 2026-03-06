import torch
import numpy as np
from skimage.measure import marching_cubes

from utils.meters import AverageMeter


def get_bound_mask(sdf, bound, mask_value=1):
    D, H, W = sdf.shape

    min_bound, max_bound = bound

    # Create voxel grid coordinates in [-1, 1] for each axis
    z = np.linspace(-1, 1, D)
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')  # shape (D, H, W)

    coords = np.stack([zz, yy, xx], axis=-1)  # shape (D, H, W, 3)

    # Create boolean mask: True where inside bounds
    in_bounds = np.all((coords >= min_bound) & (coords <= max_bound), axis=-1)

    # Apply mask
    masked_sdf = np.where(in_bounds, mask_value, 0)
    return masked_sdf

def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

def xpuk_chamfer_distance(point_cloud_1, point_cloud_2):
    # Calculate the Chamfer between transformed src and ref_clean
    dist = torch.min(square_distance(point_cloud_1, point_cloud_2), dim=-1)[0]
    # Now calculate the distance Chamfer distance between raw_ref and ref
    dist_backwards = torch.min(square_distance(point_cloud_2, point_cloud_1), dim=-1)[0]
    # Add the two calculated distances together
    chamfer_dist = torch.mean(dist, dim=1) + torch.mean(dist_backwards, dim=1)
    return chamfer_dist


def compute_L1(gt, input, pred, avg_meters, bound):
    """
    Computes all the L1 distances that need to be reported
    """

    # Check avgmeters
    if 'L1_sdf' not in avg_meters:
        avg_meters['L1_sdf'] = AverageMeter()
    if 'm_L1_sdf' not in avg_meters:
        avg_meters['m_L1_sdf'] = AverageMeter()

    # Normal SDF L1
    df_diff = float(np.mean(np.abs(gt - pred)))
    avg_meters['L1_sdf'].update(df_diff)

    # Masked SDF L1
    if bound is not None:
        mask = get_bound_mask(pred, bound)
        if mask.sum() > 0:
            l1 = np.sum(np.abs(gt - pred) * mask)
            avg_l1 = l1 / np.sum(mask)
            avg_meters['m_L1_sdf'].update(avg_l1)
        else:
            avg_meters['m_L1_sdf'].update(0.0)

def normalize_pc(pc):
    pc = pc - pc.mean(axis=0, keepdims=True)
    pc = pc / np.linalg.norm(pc, axis=1).max()
    return pc

def compute_CD(gt, input, pred, avg_meters, bound):
    """
    Computes all the CD distances that need to be reported
    """
    if 'CD_sdf' not in avg_meters:
        avg_meters['CD_sdf'] = AverageMeter()
    if 'm_CD_sdf' not in avg_meters:
        avg_meters['m_CD_sdf'] = AverageMeter()

    # Normal CD
    try:
        vertices_gt, _, _, _ = marching_cubes(gt, 0.0)
        vertices_pred, _, _, _ = marching_cubes(pred, 0.0)

        # Calculate CD
        gt_points_torch = torch.from_numpy(normalize_pc(vertices_gt.copy())).unsqueeze(0)
        pred_points_torch = torch.from_numpy(normalize_pc(vertices_pred.copy())).unsqueeze(0)
        cd = float(xpuk_chamfer_distance(gt_points_torch, pred_points_torch).cpu().numpy()[0])
        avg_meters['CD_sdf'].update(cd)
    except Exception as e:
        print("Error type:", type(e).__name__)
        avg_meters['CD_sdf'].update(avg_meters['CD_sdf'].avg)

    # Masked CD
    if bound is not None:
        mask = get_bound_mask(pred, bound)
        try:
            vert_masked_gt, _, _, _ = marching_cubes(gt * mask, 0.0)
            vert_masked_pred, _, _, _ = marching_cubes(pred * mask, 0.0)

            gt_points_torch = torch.from_numpy(normalize_pc(vert_masked_gt.copy())).unsqueeze(0)
            pred_points_torch = torch.from_numpy(normalize_pc(vert_masked_pred.copy())).unsqueeze(0)
            cd = float(xpuk_chamfer_distance(gt_points_torch, pred_points_torch).cpu().numpy()[0])
            avg_meters['m_CD_sdf'].update(cd)
        except:
            avg_meters['m_CD_sdf'].update(avg_meters['m_CD_sdf'].avg)

def compute_iou_sdf(sdf_a, sdf_b, mask=None):
    """
    This is voxel based IOU so it has precision limited by voxel grid
    """
    inside_a = sdf_a <= 0
    inside_b = sdf_b <= 0

    if mask is not None:
        intersection = np.sum((inside_a & inside_b) * mask)
        union = np.sum((inside_a | inside_b) * mask)
    else:
        intersection = np.sum(inside_a & inside_b)
        union = np.sum(inside_a | inside_b)

    iou = intersection / union if union > 0 else 0.0
    return iou


def compute_IoU(gt, input, pred, avg_meters, bound=None, antag=None, precision=False):
    """
    Computes all the IoU distances that need to be reported
    """
    if 'pred_gt_IoU' not in avg_meters:
        avg_meters['pred_gt_IoU'] = AverageMeter()
    if 'm_pred_gt_IoU' not in avg_meters:
        avg_meters['m_pred_gt_IoU'] = AverageMeter()

    # Voxelized IoU
    pred_iou = compute_iou_sdf(gt, pred)
    avg_meters['pred_gt_IoU'].update(pred_iou)

    if bound is not None:
        masked_iou = compute_iou_sdf(gt, pred, mask=get_bound_mask(pred, bound))
        avg_meters['m_pred_gt_IoU'].update(masked_iou)

    if antag is not None:

        if 'gt_antag_IoU' not in avg_meters:
            avg_meters['gt_antag_IoU'] = AverageMeter()
        if 'pred_antag_IoU' not in avg_meters:
            avg_meters['pred_antag_IoU'] = AverageMeter()

        gt_antag_iou = compute_iou_sdf(gt, antag)
        pred_antag_iou = compute_iou_sdf(pred, antag)
        avg_meters['gt_antag_IoU'].update(gt_antag_iou)
        avg_meters['pred_antag_IoU'].update(pred_antag_iou)