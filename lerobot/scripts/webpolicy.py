


# lerobot/scripts/serve_policy.py
import dataclasses
import enum
import logging
import socket
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import json
import numpy as np
import asyncio
import websockets
import time
import traceback
import msgpack_numpy

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.common.policies.factory import get_policy_class

from contextlib import nullcontext
from dataclasses import asdict, field
from pprint import pformat
from termcolor import colored
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.eval import EvalPipelineConfig
import einops
import torch
from torch import Tensor
def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()
    return return_observations

class WebPolicyServer:
    def __init__(self, policy: VQBeTPolicy, device: str, host: str = "0.0.0.0", port: int = 8000):
        """初始化策略服务器"""
        self.policy = policy
        self.policy.reset()
        self.device = torch.device(device)
        self.host = host
        self.port = port
        self.server = None
        logging.info(f"PolicyServer initialized with device={device}, host={host}, port={port}")
        logging.info(f"Policy type: {type(policy).__name__}")
        
        # 记录模型输入输出特征
        logging.info(f"Policy input features: {policy.config.input_features}")
        logging.info(f"Policy output features: {policy.config.output_features}")

    def server_forever(self):
        asyncio.run(self.run())
    async def run(self):
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                print(f"Received observation: ")
                for key,value in obs.items():
                    obs[key]=torch.from_numpy(value).to(self.device)
                    print(f"shape of {key}:{obs[key].shape}")
                print(f"observation device: {obs['observation.image_0'].device}")
                action = self.policy.select_action(obs)
                print(f"action: {action.shape}")
                #time.sleep()

                packed_action = msgpack_numpy.packb(action.cpu().numpy(), use_bin_type=True)
                await websocket.send(packed_action)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
"""

Before stack, queue contents:
  Queue 'observation.images' has 5 items:
    Item 0: shape=torch.Size([1, 3, 480, 480]), dtype=torch.float32
    Item 1: shape=torch.Size([1, 3, 480, 480]), dtype=torch.float32
    Item 2: shape=torch.Size([1, 3, 480, 480]), dtype=torch.float32
    Item 3: shape=torch.Size([1, 3, 480, 480]), dtype=torch.float32
    Item 4: shape=torch.Size([1, 3, 480, 480]), dtype=torch.float32
  Queue 'observation.state' has 5 items:
    Item 0: shape=torch.Size([7]), dtype=torch.float32
    Item 1: shape=torch.Size([7]), dtype=torch.float32
    Item 2: shape=torch.Size([7]), dtype=torch.float32
    Item 3: shape=torch.Size([7]), dtype=torch.float32
    Item 4: shape=torch.Size([7]), dtype=torch.float32
after stack observation.state: torch.Size([7, 5])
after stack observation.images: torch.Size([1, 5, 3, 480, 480])
"""