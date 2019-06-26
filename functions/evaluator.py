import os
from logging import Logger

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from functions.base import CheckpointRunner
from models import SMPL, CMR
from models.geometric_layers import orthographic_projection
from utils.average_meter import AverageMeter
from utils.imutils import uncrop
from utils.mesh import Mesh
from utils.part_utils import PartRenderer
from utils.pose_utils import reconstruction_error
from utils.renderer import Renderer, visualize_batch_recon_and_keypoints


class Evaluator(CheckpointRunner):

    def __init__(self, options, logger: Logger, dataset, shared_model=None):
        self.annot_path = None
        self.eval_pose = self.eval_shape = self.eval_masks = self.eval_parts = False
        if dataset == 'h36m-p1' or dataset == 'h36m-p2':
            self.eval_pose = True
        elif dataset == 'up-3d':
            self.eval_shape = True
        elif dataset == 'lsp':
            self.eval_masks = True
            self.eval_parts = True
            self.annot_path = config.DATASET_FOLDERS['upi-s1h']
        else:
            raise ValueError("Unsupported Dataset")
        self.dataset_name = dataset
        super().__init__(options, logger, dataset=dataset, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.mesh = Mesh()
        self.faces = self.mesh.faces.cuda()

        if shared_model is not None:
            self.cmr = shared_model
        else:
            # create GraphCNN + SMPLRegressor
            self.cmr = CMR(self.mesh,
                           num_layers=self.options.model.num_layers,
                           num_channels=self.options.model.num_channels)
            self.cmr = torch.nn.DataParallel(self.cmr, device_ids=self.gpus).cuda()
        self.smpl = SMPL().cuda()

        # Regressor for H36m joints
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

        # Renderer for visualization
        self.renderer = Renderer(faces=self.smpl.faces.cpu().detach())
        self.part_renderer = PartRenderer()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0

    def models_dict(self):
        return {'graph_cnn': self.cmr.module.graph_cnn,
                'smpl_param_regressor': self.cmr.module.smpl_param_regressor}

    def evaluate_pose(self, pred_vertices, pred_vertices_smpl, gt_pose_3d):
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).cuda()

        # Get 14 ground truth joints
        gt_keypoints_3d = gt_pose_3d.cuda()
        gt_keypoints_3d = gt_keypoints_3d[:, config.J24_TO_J14, :-1]

        # Get 14 predicted joints from the non-parametic mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, config.H36M_TO_J14, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices_smpl)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, config.H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        # Compute error metrics

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
            dim=-1).cpu().numpy()
        self.mpjpe.update(error)
        self.mpjpe_smpl.update(error_smpl)

        # Reconstuction_error
        r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                       reduction=None)
        r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                            reduction=None)
        self.recon_err.update(r_error)
        self.recon_err_smpl.update(r_error_smpl)

    def evaluate_shape(self, pred_vertices, pred_vertices_smpl, gt_vertices):
        se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        se_smpl = torch.sqrt(((pred_vertices_smpl - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        self.shape_err.update(se)
        self.shape_err_smpl.update(se_smpl)

    def evaluate_mask_and_parts_for_lsp(self, pred_vertices, pred_camera, center, scale, orig_shape,
                                        maskname, partname):
        mask, parts = self.part_renderer(pred_vertices, pred_camera)
        center = center.cpu().numpy()
        scale = scale.cpu().numpy()
        # Dimensions of original image
        orig_shape = orig_shape.cpu().numpy()
        curr_batch_size = pred_vertices.size(0)
        if self.eval_masks:
            for i in range(curr_batch_size):
                # After rendering, convert image back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(self.annot_path, maskname[i]), 0) > 0

                # Evaluation consistent with the original UP-3D code
                self.accuracy.update((gt_mask == pred_mask).sum() / np.prod(np.array(gt_mask.shape)))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    self.tp[c] += (cgt & cpred).sum()
                    self.fp[c] += (~cgt & cpred).sum()
                    self.fn[c] += (cgt & ~cpred).sum()

        if self.eval_parts:
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(self.annot_path, partname[i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                    cgt = gt_parts == c
                    cpred = pred_parts == c
                    cpred[gt_parts == 255] = 0
                    self.parts_tp[c] += (cgt & cpred).sum()
                    self.parts_fp[c] += (~cgt & cpred).sum()
                    self.parts_fn[c] += (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                self.parts_accuracy.update((gt_parts == pred_parts).sum() / np.prod(np.array(gt_parts.shape)))

    @staticmethod
    def f1_mean(tp, fp, fn):
        return (2 * tp / (2 * tp + fp + fn)).mean()

    def evaluate_step(self, input_batch):
        self.cmr.eval()

        # Get ground truth
        gt_pose = input_batch['pose'].cuda()
        gt_betas = input_batch['betas'].cuda()
        gt_vertices = self.smpl(gt_pose, gt_betas)
        images = input_batch['img'].cuda()

        # Run inference
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, pred_camera, pred_rotmat, pred_betas = self.cmr(images)

        if self.eval_pose:
            self.evaluate_pose(pred_vertices, pred_vertices_smpl, input_batch['pose_3d'])
        if self.eval_shape:
            self.evaluate_shape(pred_vertices, pred_vertices_smpl, gt_vertices)
        if self.eval_masks or self.eval_parts:
            self.evaluate_mask_and_parts_for_lsp(pred_vertices, pred_camera, input_batch['center'],
                                                 input_batch['scale'], input_batch['orig_shape'],
                                                 input_batch['maskname'], input_batch['partname'])

        return pred_vertices, pred_vertices_smpl, pred_camera, pred_rotmat, pred_betas

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        self.logger.info("Running evaluations on [%s]" % self.dataset_name)
        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle)

        # Initialize average meters
        if self.eval_pose:
            # Pose metrics: MPJPE and Reconstruction error for the non-parametric and parametric shapes
            self.mpjpe = AverageMeter(1000)
            self.recon_err = AverageMeter(1000)
            self.mpjpe_smpl = AverageMeter(1000)
            self.recon_err_smpl = AverageMeter(1000)
        if self.eval_shape:
            # Mean per-vertex error
            self.shape_err = AverageMeter(1000)
            self.shape_err_smpl = AverageMeter(1000)
        if self.eval_masks:
            self.accuracy = AverageMeter()
            # True positive, false positive and false negative
            self.tp, self.fp, self.fn = np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))
        if self.eval_parts:
            self.parts_accuracy = AverageMeter()
            self.parts_tp, self.parts_fp, self.parts_fn = np.zeros((7, 1)), np.zeros((7, 1)), np.zeros((7, 1))

        # Iterate over all batches in an epoch
        for step, batch in enumerate(test_data_loader):
            # Run training step
            out = self.evaluate_step(batch)

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, *out)

            # add later to log at step 0
            self.evaluate_step_count += 1

    def evaluate_summaries(self, input_batch, pred_vertices, pred_vertices_smpl, pred_camera, pred_rotmat, pred_betas):
        message = "Test [%s] Step %06d/%06d " % (self.dataset_name, self.evaluate_step_count,
                                                 len(self.dataset) // (
                                                             self.options.num_gpus * self.options.test.batch_size))

        if self.eval_pose:
            message += "MPJPE-NonP %s, Recon-NonP %s, MPJPE-Param %s, Recon-Param %s " % (
                self.mpjpe, self.recon_err, self.mpjpe_smpl, self.recon_err_smpl
            )
        if self.eval_shape:
            message += "Shape-NonP %s, Shape-Param %s " % (self.shape_err, self.shape_err_smpl)
        if self.eval_masks:
            message += "Accuracy %s, F1 %.6f " % (self.accuracy, self.f1_mean(self.tp, self.fp, self.fn))
        if self.eval_parts:
            message += "Parts Accuracy %s, Parts F1 %.6f " % (
            self.parts_accuracy, self.f1_mean(self.parts_tp, self.parts_fp, self.parts_fn))
        self.logger.info(message)

        pred_keypoints_3d = self.smpl.get_joints(pred_vertices)
        pred_keypoints_2d = orthographic_projection(pred_keypoints_3d, pred_camera)[:, :, :2]
        pred_keypoints_3d_smpl = self.smpl.get_joints(pred_vertices_smpl)
        pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, pred_camera)[:, :, :2]

        rend_imgs, rend_imgs_smpl = visualize_batch_recon_and_keypoints(4, self.renderer, input_batch["img_orig"],
                                                                        pred_vertices, pred_vertices_smpl,
                                                                        pred_camera, pred_keypoints_2d,
                                                                        pred_keypoints_2d_smpl,
                                                                        input_batch['keypoints'])

        # Save results in Tensorboard
        self.summary_writer.add_image('eval_imgs', rend_imgs, self.step_count)
        self.summary_writer.add_image('eval_imgs_smpl', rend_imgs_smpl, self.step_count)
