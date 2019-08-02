import os
import random
from logging import Logger

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.base import CheckpointRunner
from models.p2m import P2MModel
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer


class Predictor(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.gpu_inference = self.options.num_gpus > 0
        if self.gpu_inference == 0:
            raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                      "might be fixed in the future.")
            # self.logger.warning("Render part would be disabled since you are using CPU. "
            #                     "Neural renderer requires GPU to run. Please use other softwares "
            #                     "or packages to view .obj file generated.")

        if self.options.model.name == "pixel2mesh":
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # create model
            self.model = P2MModel(self.options.model, self.ellipsoid,
                                  self.options.dataset.camera_f, self.options.dataset.camera_c,
                                  self.options.dataset.mesh_pos)
            if self.gpu_inference:
                self.model.cuda()
                # create renderer
                self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                             self.options.dataset.mesh_pos)
        else:
            raise NotImplementedError("Currently the predictor only supports pixel2mesh")

    def models_dict(self):
        return {'model': self.model}

    def predict_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            images = input_batch['images']
            out = self.model(images)
            self.save_inference_results(input_batch, out)

    def predict(self):
        self.logger.info("Running predictions...")

        predict_data_loader = DataLoader(self.dataset,
                                         batch_size=self.options.test.batch_size,
                                         pin_memory=self.options.pin_memory,
                                         collate_fn=self.dataset_collate_fn)

        for step, batch in enumerate(predict_data_loader):
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            if self.gpu_inference:
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.predict_step(batch)

    def save_inference_results(self, inputs, outputs):
        if self.options.model.name == "pixel2mesh":
            batch_size = inputs["images"].size(0)
            for i in range(batch_size):
                basename, ext = os.path.splitext(inputs["filepath"][i])
                mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
                verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
                for k, vert in enumerate(verts):
                    meshname = basename + ".%d.obj" % (k + 1)
                    vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                    mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                    np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")

                if self.gpu_inference:
                    # generate gif here

                    color_repo = ['light_blue', 'purple', 'orange', 'light_yellow']

                    rot_degree = 10
                    rot_radius = rot_degree / 180 * np.pi
                    rot_matrix = np.array([
                        [np.cos(rot_radius), 0, -np.sin(rot_radius)],
                        [0., 1., 0.],
                        [np.sin(rot_radius), 0, np.cos(rot_radius)]
                    ])
                    writer = imageio.get_writer(basename + ".gif", mode='I')
                    color = random.choice(color_repo)
                    for _ in tqdm(range(360 // rot_degree), desc="Rendering sample %d" % i):
                        image = inputs["images_orig"][i].cpu().numpy()
                        ret = image
                        for k, vert in enumerate(verts):
                            vert = rot_matrix.dot((vert - mesh_center).T).T + mesh_center
                            rend_result = self.renderer.visualize_reconstruction(None,
                                                                                 vert + \
                                                                                 np.array(
                                                                                     self.options.dataset.mesh_pos),
                                                                                 self.ellipsoid.faces[k],
                                                                                 image,
                                                                                 mesh_only=True,
                                                                                 color=color)
                            ret = np.concatenate((ret, rend_result), axis=2)
                            verts[k] = vert
                        ret = np.transpose(ret, (1, 2, 0))
                        writer.append_data((255 * ret).astype(np.uint8))
                    writer.close()
