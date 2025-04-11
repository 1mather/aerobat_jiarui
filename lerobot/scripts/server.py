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

from lerobot.scripts.webpolicy import WebPolicyServer


class EnvMode(enum.Enum):
    """支持的环境类型"""
    FRANKA = "franka"
    XARM = "xarm"
    UR5 = "ur5"


HOST = "127.0.0.1"
PORT = 8000



# 预设配置映射
PRESETS = {
    "franka_default": {
        "env": EnvMode.FRANKA,
        "policy": {
            "config_path": "configs/vqbet/franka_default.yaml",
            "model_path": "checkpoints/vqbet/franka_default",
            "model_type": "vqbet"
        }
    },
    "xarm_default": {
        "env": EnvMode.XARM,
        "policy": {
            "config_path": "configs/vqbet/xarm_default.yaml",
            "model_path": "checkpoints/vqbet/xarm_default",
            "model_type": "vqbet"
        }
    },
    "ur5_default": {
        "env": EnvMode.UR5,
        "policy": {
            "config_path": "configs/vqbet/ur5_default.yaml",
            "model_path": "checkpoints/vqbet/ur5_default",
            "model_type": "vqbet"
        }
    }
}




    

    # def _process_observation(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     """处理观测数据"""
    #     processed_data = {}
        
    #     try:
    #         # 检查观测数据格式
    #         if not isinstance(data, dict):
    #             raise ValueError(f"Expected dictionary input, got {type(data).__name__}")
            
    #         # 处理图像数据
    #         for i in range(5):
    #             key = f"observation.image_{i}"
    #             img_key = f"image_{i}"
                
    #             if img_key in data:
    #                 img_data = data[img_key]
    #                 logging.debug(f"Processing image_{i}, type: {type(img_data)}")
                    
    #                 # 验证图像数据
    #                 if not isinstance(img_data, list):
    #                     raise ValueError(f"Image data for {img_key} is not a list, got {type(img_data).__name__}")
                    
    #                 # 转换为张量
    #                 try:
    #                     img_tensor = torch.tensor(img_data, device=self.device, dtype=torch.float32)
    #                     logging.debug(f"Converted {img_key} to tensor with shape {img_tensor.shape}")
                        
    #                     # 图像应该是3D的 [H, W, C] 或 [C, H, W]
    #                     if img_tensor.dim() != 3:
    #                         raise ValueError(f"Image tensor for {img_key} has {img_tensor.dim()} dimensions, expected 3")
                        
    #                     processed_data[key] = img_tensor
    #                 except Exception as e:
    #                     logging.exception(f"Error converting {img_key} to tensor: {e}")
    #                     raise ValueError(f"Failed to convert {img_key} to tensor: {str(e)}")
            
    #         # 处理状态数据
    #         if "state" in data:
    #             state_data = data["state"]
    #             logging.debug(f"Processing state, type: {type(state_data)}")
                
    #             # 验证状态数据
    #             if not isinstance(state_data, list):
    #                 raise ValueError(f"State data is not a list, got {type(state_data).__name__}")
                
    #             # 转换为张量
    #             try:
    #                 state_tensor = torch.tensor(state_data, device=self.device, dtype=torch.float32)
    #                 logging.debug(f"Converted state to tensor with shape {state_tensor.shape}")
                    
    #                 # 状态应该是1D的
    #                 if state_tensor.dim() != 1:
    #                     raise ValueError(f"State tensor has {state_tensor.dim()} dimensions, expected 1")
                    
    #                 # 检查状态向量长度
    #                 expected_length = next(
    #                     (feature.shape[0] for name, feature in self.policy.config.input_features.items() 
    #                      if name == "observation.state"),
    #                     None
    #                 )
    #                 if expected_length and state_tensor.shape[0] != expected_length:
    #                     raise ValueError(f"State vector has length {state_tensor.shape[0]}, expected {expected_length}")
                    
    #                 processed_data["observation.state"] = state_tensor
    #             except Exception as e:
    #                 logging.exception(f"Error converting state to tensor: {e}")
    #                 raise ValueError(f"Failed to convert state to tensor: {str(e)}")
            
    #         # 验证至少有一个观测数据
    #         if not processed_data:
    #             raise ValueError("No valid observation data found in the input")
            
    #         return processed_data
        
    #     except Exception as e:
    #         logging.exception(f"Error in _process_observation: {e}")
    #         raise

    # def _prepare_model_input(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """准备模型输入"""
    #     try:
    #         # 验证观测数据
    #         if not observation:
    #             raise ValueError("Empty observation dictionary")
            
    #         # 添加批次维度
    #         result = {}
    #         for k, v in observation.items():
    #             if not isinstance(v, torch.Tensor):
    #                 logging.warning(f"Non-tensor value for key '{k}': {type(v).__name__}")
    #                 continue
                
    #             # 图像通常已经是4D的 [B, C, H, W]
    #             if v.dim() < 4 and k.startswith("observation.image_"):
    #                 # 如果是 [H, W, C]，先转换为 [C, H, W]
    #                 if v.dim() == 3 and v.shape[2] == 3:
    #                     v = v.permute(2, 0, 1)
    #                     logging.debug(f"Permuted {k} from [H,W,C] to [C,H,W]")
                    
    #                 # 添加批次维度
    #                 v = v.unsqueeze(0)
    #                 logging.debug(f"Added batch dimension to {k}, new shape: {v.shape}")
                
    #             # 状态通常是1D的 [D]，添加批次维度变为 [B, D]
    #             elif v.dim() == 1:
    #                 v = v.unsqueeze(0)
    #                 logging.debug(f"Added batch dimension to {k}, new shape: {v.shape}")
                
    #             result[k] = v
            
    #         # 验证结果
    #         if not result:
    #             raise ValueError("Failed to prepare any valid model inputs")
            
    #         return result
        
    #     except Exception as e:
    #         logging.exception(f"Error in _prepare_model_input: {e}")
    #         raise

    # def _process_prediction(self, prediction: Dict[str, torch.Tensor]) -> tuple:
    #     """处理模型预测结果"""
    #     try:
    #         # 验证预测结果
    #         if not isinstance(prediction, dict):
    #             raise ValueError(f"Expected dictionary prediction, got {type(prediction).__name__}")
            
    #         if "action" not in prediction:
    #             raise ValueError(f"Missing 'action' in prediction. Available keys: {list(prediction.keys())}")
            
    #         # 获取动作预测
    #         action = prediction["action"]
    #         logging.debug(f"Raw action shape: {action.shape}")
            
    #         # 验证动作格式
    #         if not isinstance(action, torch.Tensor):
    #             raise ValueError(f"Action is not a tensor, got {type(action).__name__}")
            
    #         # 移除批次维度
    #         if action.dim() > 1:
    #             action = action[0]  # 取第一个批次的结果
    #             logging.debug(f"Removed batch dimension, action shape: {action.shape}")
            
    #         # 验证动作维度
    #         expected_dim = next(
    #             (feature.shape[0] for name, feature in self.policy.config.output_features.items() 
    #              if name == "action"),
    #             None
    #         )
    #         if expected_dim and action.shape[0] != expected_dim:
    #             raise ValueError(f"Action has {action.shape[0]} dimensions, expected {expected_dim}")
            
    #         # 确保至少有7个元素（3个位置 + 3个欧拉角 + 1个夹持器）
    #         if action.shape[0] < 7:
    #             raise ValueError(f"Action has only {action.shape[0]} elements, need at least 7")
            
    #         # 分解动作为位置、欧拉角和夹持器状态
    #         position = action[:3]
    #         euler = action[3:6]
    #         gripper = action[6]
    #         view_index = 0  # 默认视角
            
    #         # 将结果移到CPU并转换为numpy
    #         position_np = position.cpu().numpy()
    #         euler_np = euler.cpu().numpy()
    #         gripper_value = gripper.cpu().item()
            
    #         logging.debug(f"Processed prediction: position={position_np}, euler={euler_np}, gripper={gripper_value}")
            
    #         return position_np, euler_np, gripper_value, view_index
        
    #     except Exception as e:
    #         logging.exception(f"Error in _process_prediction: {e}")
    #         raise

    # async def start(self):
    #     """异步启动服务器"""
    #     try:
    #         self.server = await self._setup_server()
    #         logging.info(f"Server started successfully on {self.host}:{self.port}")
    #     except Exception as e:
    #         logging.exception(f"Failed to start server: {e}")
    #         raise

    # def serve_forever(self):
    #     """同步启动服务器并保持运行"""
    #     logging.info("Starting server in serve_forever mode")
    #     try:
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
            
    #         try:
    #             logging.info("Setting up event loop")
    #             loop.run_until_complete(self.start())
    #             logging.info("Server running, entering event loop")
    #             loop.run_forever()
    #         except KeyboardInterrupt:
    #             logging.info("Server stopped by keyboard interrupt")
    #         except Exception as e:
    #             logging.exception(f"Error in event loop: {e}")
    #             raise
    #         finally:
    #             logging.info("Closing event loop")
    #             loop.close()
    #     except Exception as e:
    #         logging.exception(f"Fatal error in serve_forever: {e}")
    #         raise



@parser.wrap()
def server_main(cfg: EvalPipelineConfig):
    """主函数"""
    # 设置日志
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    #env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()
    # 确定设备
    device = cfg.policy.device
    # 创建上下文管理器
    ctx = torch.autocast(device_type=device if device != "cpu" else "cpu") if cfg.policy.use_amp else nullcontext()


    with torch.no_grad(), ctx:
        # 获取主机信息
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")
        
        # 创建并启动服务器
        server = WebPolicyServer(
            policy=policy,
            device=device,
            host=HOST,
            port=PORT

        )
        
        try:
            server.server_forever()
            
        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        except Exception as e:
            logging.error(f"Server error: {e}")
            import traceback
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    init_logging()
    server_main()

"""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个 GPU

# 启动服务器
python lerobot/scripts/server.py --policy.path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model --policy.config_path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model/config.json



eval:
usage: server.py [-h] [--config_path str] [--env str] [--env.type {aloha,pusht,xarm}] [--env.task str] [--env.fps str]
                 [--env.features str] [--env.features_map str] [--env.episode_length str] [--env.obs_type str]
                 [--env.render_mode str] [--env.visualization_width str] [--env.visualization_height str] [--eval str]
                 [--eval.n_episodes str] [--eval.batch_size str] [--eval.use_async_envs str] [--policy str]
                 [--policy.type {act,diffusion,pi0,tdmpc,vqbet,pi0fast}]
                 [--policy.replace_final_stride_with_dilation str] [--policy.pre_norm str] [--policy.dim_model str]
                 [--policy.n_heads str] [--policy.dim_feedforward str] [--policy.feedforward_activation str]
                 [--policy.n_encoder_layers str] [--policy.n_decoder_layers str] [--policy.use_vae str]
                 [--policy.n_vae_encoder_layers str] [--policy.temporal_ensemble_coeff str] [--policy.kl_weight str]
                 [--policy.optimizer_lr_backbone str] [--policy.drop_n_last_frames str]
                 [--policy.use_separate_rgb_encoder_per_camera str] [--policy.down_dims str]
                 [--policy.kernel_size str] [--policy.n_groups str] [--policy.diffusion_step_embed_dim str]
                 [--policy.use_film_scale_modulation str] [--policy.noise_scheduler_type str]
                 [--policy.num_train_timesteps str] [--policy.beta_schedule str] [--policy.beta_start str]
                 [--policy.beta_end str] [--policy.prediction_type str] [--policy.clip_sample str]
                 [--policy.clip_sample_range str] [--policy.num_inference_steps str]
                 [--policy.do_mask_loss_for_padding str] [--policy.scheduler_name str] [--policy.num_steps str]
                 [--policy.attention_implementation str] [--policy.train_expert_only str]
                 [--policy.train_state_proj str] [--policy.n_action_repeats str] [--policy.horizon str]
                 [--policy.image_encoder_hidden_dim str] [--policy.state_encoder_hidden_dim str]
                 [--policy.latent_dim str] [--policy.q_ensemble_size str] [--policy.mlp_dim str]
                 [--policy.discount str] [--policy.use_mpc str] [--policy.cem_iterations str] [--policy.max_std str]
                 [--policy.min_std str] [--policy.n_gaussian_samples str] [--policy.n_pi_samples str]
                 [--policy.uncertainty_regularizer_coeff str] [--policy.n_elites str]
                 [--policy.elite_weighting_temperature str] [--policy.gaussian_mean_momentum str]
                 [--policy.max_random_shift_ratio str] [--policy.reward_coeff str] [--policy.expectile_weight str]
                 [--policy.value_coeff str] [--policy.consistency_coeff str] [--policy.advantage_scaling str]
                 [--policy.pi_coeff str] [--policy.temporal_decay_coeff str] [--policy.target_model_momentum str]
                 [--policy.n_action_pred_token str] [--policy.action_chunk_size str] [--policy.vision_backbone str]
                 [--policy.crop_shape str] [--policy.crop_is_random str] [--policy.pretrained_backbone_weights str]
                 [--policy.use_group_norm str] [--policy.spatial_softmax_num_keypoints str]
                 [--policy.n_vqvae_training_steps str] [--policy.vqvae_n_embed str] [--policy.vqvae_embedding_dim str]
                 [--policy.vqvae_enc_hidden_dim str] [--policy.gpt_block_size str] [--policy.gpt_input_dim str]
                 [--policy.gpt_output_dim str] [--policy.gpt_n_layer str] [--policy.gpt_n_head str]
                 [--policy.gpt_hidden_dim str] [--policy.dropout str] [--policy.mlp_hidden_dim str]
                 [--policy.offset_loss_weight str] [--policy.primary_code_loss_weight str]
                 [--policy.secondary_code_loss_weight str] [--policy.bet_softmax_temperature str]
                 [--policy.sequentially_select str] [--policy.optimizer_vqvae_lr str]
                 [--policy.optimizer_vqvae_weight_decay str] [--policy.input_camera str] [--policy.n_obs_steps str]
                 [--policy.normalization_mapping str] [--policy.input_features str] [--policy.output_features str]
                 [--policy.device str] [--policy.use_amp str] [--policy.chunk_size str] [--policy.n_action_steps str]
                 [--policy.max_state_dim str] [--policy.max_action_dim str] [--policy.resize_imgs_with_padding str]
                 [--policy.interpolate_like_pi str] [--policy.empty_cameras str] [--policy.adapt_to_pi_aloha str]
                 [--policy.use_delta_joint_actions_aloha str] [--policy.tokenizer_max_length str]
                 [--policy.proj_width str] [--policy.max_decoding_steps str] [--policy.fast_skip_tokens str]
                 [--policy.max_input_seq_len str] [--policy.use_cache str] [--policy.freeze_vision_encoder str]
                 [--policy.freeze_lm_head str] [--policy.optimizer_lr str] [--policy.optimizer_betas str]
                 [--policy.optimizer_eps str] [--policy.optimizer_weight_decay str]
                 [--policy.scheduler_warmup_steps str] [--policy.scheduler_decay_steps str]
                 [--policy.scheduler_decay_lr str] [--policy.checkpoint_path str] [--policy.padding_side str]
                 [--policy.precision str] [--policy.grad_clip_norm str] [--policy.relaxed_action_decoding str]
                 [--output_dir str] [--job_name str] [--seed str]


                 python eval.py --policy.path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model
"""