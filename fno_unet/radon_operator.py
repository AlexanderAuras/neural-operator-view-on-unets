from math import sqrt
from typing import Any, override

import astra
import torch
import torch.utils.data


class RadonForward(torch.autograd.Function):
    @staticmethod
    @override
    def forward(ctx: Any, img: torch.Tensor, det_count: int, angles: torch.Tensor) -> torch.Tensor:
        ctx.img_shape = img.shape
        ctx.det_count = det_count
        ctx.save_for_backward(angles)
        img_geom_conf = astra.create_vol_geom((img.shape[-1], img.shape[-2]))
        proj_geom_conf = astra.create_proj_geom("parallel", sqrt(img.shape[-1] ** 2 + img.shape[-2] ** 2) / det_count, det_count, angles.detach().to("cpu").numpy())
        img_data_id = astra.data2d.create("-vol", img_geom_conf)
        sino_data_id = astra.data2d.create("-sino", proj_geom_conf)
        algo_conf = astra.astra_dict("FP_CUDA")
        algo_conf["VolumeDataId"] = img_data_id
        algo_conf["ProjectionDataId"] = sino_data_id
        algo_id = astra.algorithm.create(algo_conf)
        flat_batch_img = img.flatten(end_dim=-3)
        flat_batch_sino = torch.zeros((flat_batch_img.shape[0], angles.shape[0], det_count), dtype=img.dtype, device=img.device)
        for i in range(flat_batch_img.shape[0]):
            astra.data2d.store(img_data_id, flat_batch_img[i].detach().to("cpu").numpy())
            astra.algorithm.run(algo_id)
            flat_batch_sino[i] = torch.from_numpy(astra.data2d.get(sino_data_id)).to(img.device)
        astra.algorithm.delete(algo_id)
        astra.data2d.delete(sino_data_id)
        astra.data2d.delete(img_data_id)
        return flat_batch_sino.reshape(*img.shape[:-2], *flat_batch_sino.shape[-2:])

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None, None]:
        img_shape = ctx.img_shape
        det_count = ctx.det_count
        (angles,) = ctx.saved_tensors
        sino = grad_outputs[0]
        img_geom_conf = astra.create_vol_geom((img_shape[-1], img_shape[-2]))
        proj_geom_conf = astra.create_proj_geom("parallel", sqrt(img_shape[-1] ** 2 + img_shape[-2] ** 2) / det_count, det_count, angles.detach().to("cpu").numpy())
        img_data_id = astra.data2d.create("-vol", img_geom_conf)
        sino_data_id = astra.data2d.create("-sino", proj_geom_conf)
        proj_id = astra.create_projector("cuda", proj_geom_conf, img_geom_conf)
        algo_conf = astra.astra_dict("BP_CUDA")
        algo_conf["ProjectorId"] = proj_id
        algo_conf["ProjectionDataId"] = sino_data_id
        algo_conf["ReconstructionDataId"] = img_data_id
        algo_id = astra.algorithm.create(algo_conf)
        flat_batch_sino = sino.flatten(end_dim=-3)
        flat_batch_img = torch.zeros((flat_batch_sino.shape[0], *img_shape[-2:]), dtype=sino.dtype, device=sino.device)
        for i in range(flat_batch_img.shape[0]):
            astra.data2d.store(sino_data_id, flat_batch_sino[i].detach().to("cpu").numpy())
            astra.algorithm.run(algo_id)
            flat_batch_img[i] = torch.from_numpy(astra.data2d.get(img_data_id)).to(sino.device)
        astra.algorithm.delete(algo_id)
        astra.projector.delete(proj_id)
        astra.data2d.delete(sino_data_id)
        astra.data2d.delete(img_data_id)
        return flat_batch_img.reshape(*sino.shape[:-2], *flat_batch_img.shape[-2:]), None, None


class RadonBackward(torch.autograd.Function):
    @staticmethod
    @override
    def forward(ctx: Any, sino: torch.Tensor, img_shape: torch.Size, det_count: int, angles: torch.Tensor) -> torch.Tensor:
        ctx.det_count = det_count
        ctx.save_for_backward(angles)
        img_geom_conf = astra.create_vol_geom((img_shape[-1], img_shape[-2]))
        proj_geom_conf = astra.create_proj_geom("parallel", sqrt(img_shape[-1] ** 2 + img_shape[-2] ** 2) / det_count, det_count, angles.detach().to("cpu").numpy())
        img_data_id = astra.data2d.create("-vol", img_geom_conf)
        sino_data_id = astra.data2d.create("-sino", proj_geom_conf)
        proj_id = astra.create_projector("cuda", proj_geom_conf, img_geom_conf)
        algo_conf = astra.astra_dict("FBP_CUDA")
        algo_conf["ProjectorId"] = proj_id
        algo_conf["ProjectionDataId"] = sino_data_id
        algo_conf["ReconstructionDataId"] = img_data_id
        algo_id = astra.algorithm.create(algo_conf)
        flat_batch_sino = sino.flatten(end_dim=-3)
        flat_batch_img = torch.zeros((flat_batch_sino.shape[0], *img_shape[-2:]), dtype=sino.dtype, device=sino.device)
        for i in range(flat_batch_img.shape[0]):
            astra.data2d.store(sino_data_id, flat_batch_sino[i].detach().to("cpu").numpy())
            astra.algorithm.run(algo_id)
            flat_batch_img[i] = torch.from_numpy(astra.data2d.get(img_data_id)).to(sino.device)
        astra.algorithm.delete(algo_id)
        astra.projector.delete(proj_id)
        astra.data2d.delete(sino_data_id)
        astra.data2d.delete(img_data_id)
        return flat_batch_img.reshape(*sino.shape[:-2], *flat_batch_img.shape[-2:])

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None, None, None]:
        det_count = ctx.det_count
        (angles,) = ctx.saved_tensors
        img = grad_outputs[0]
        img_geom_conf = astra.create_vol_geom((img.shape[-1], img.shape[-2]))
        proj_geom_conf = astra.create_proj_geom("parallel", sqrt(img.shape[-1] ** 2 + img.shape[-2] ** 2) / det_count, det_count, angles.detach().to("cpu").numpy())
        img_data_id = astra.data2d.create("-vol", img_geom_conf)
        sino_data_id = astra.data2d.create("-sino", proj_geom_conf)
        algo_conf = astra.astra_dict("FP_CUDA")
        algo_conf["VolumeDataId"] = img_data_id
        algo_conf["ProjectionDataId"] = sino_data_id
        algo_id = astra.algorithm.create(algo_conf)
        flat_batch_img = img.flatten(end_dim=-3)
        flat_batch_sino = torch.zeros((flat_batch_img.shape[0], angles.shape[0], det_count), dtype=img.dtype, device=img.device)
        for i in range(flat_batch_img.shape[0]):
            astra.data2d.store(img_data_id, flat_batch_img[i].detach().to("cpu").numpy())
            astra.algorithm.run(algo_id)
            flat_batch_sino[i] = torch.from_numpy(astra.data2d.get(sino_data_id)).to(img.device)
        astra.algorithm.delete(algo_id)
        astra.data2d.delete(sino_data_id)
        astra.data2d.delete(img_data_id)
        return flat_batch_sino.reshape(*img.shape[:-2], *flat_batch_sino.shape[-2:]), None, None, None
