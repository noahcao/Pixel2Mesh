from logging import Logger

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from models.layers.chamfer_wrapper import ChamferDist
from models.p2m import P2MModel
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer


class Evaluator(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # create ellipsoid
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = P2MModel(self.options.model, self.ellipsoid,
                                  self.options.dataset.camera_f, self.options.dataset.camera_c,
                                  self.options.dataset.mesh_pos)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Renderer for visualization
        self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                     self.options.dataset.mesh_pos)

        # Initialize distance module
        self.chamfer = ChamferDist()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, gt_length):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = gt_points.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length_i = min(gt_length[i].cpu().item(), self.options.dataset.shapenet.num_points)
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i, :gt_length_i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            self.chamfer_distance.update(np.mean(d1) + np.mean(d2))
            self.f1_tau.update(self.evaluate_f1(d1, d2, pred_length, gt_length_i, 1E-4))
            self.f1_2tau.update(self.evaluate_f1(d1, d2, pred_length, gt_length_i, 2E-4))

    def evaluate_step(self, input_batch):
        self.model.eval()

        # Get ground truth
        images = input_batch['images']
        gt_points = input_batch['points']
        gt_length = input_batch['length']

        # Run inference
        with torch.no_grad():
            out = self.model(images)
            pred_vertices = out["pred_coord"][0]

            self.evaluate_chamfer_and_f1(pred_vertices, gt_points, gt_length)

        return out

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        self.logger.info("Running evaluations...")

        # clear evaluate_step_count, but keep total count uncleared
        self.evaluate_step_count = 0

        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle)

        self.chamfer_distance = AverageMeter()
        self.f1_tau = AverageMeter()
        self.f1_2tau = AverageMeter()

        # Iterate over all batches in an epoch
        for step, batch in enumerate(test_data_loader):
            # Send input to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Run evaluation step
            out = self.evaluate_step(batch)

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)

            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1

        for key, val in self.get_result_summary().items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
            self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)

    def get_result_summary(self):
        result_dict = {
            "cd": self.chamfer_distance,
            "f1_tau": self.f1_tau,
            "f1_2tau": self.f1_2tau,
        }
        return result_dict

    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                 len(self.dataset) // (
                                                         self.options.num_gpus * self.options.test.batch_size),
                                                               self.total_step_count,) \
            + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                         for key, val in self.get_result_summary().items()]))

        # Do visualization for the first 2 images of the batch
        render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
        self.summary_writer.add_image("eval_render_mesh", render_mesh, self.total_step_count)

