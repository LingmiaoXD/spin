import copy
import datetime
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.imputers import Imputer
from tsl.nn.metrics import MaskedMetric, MaskedMAE, MaskedMSE, MaskedMRE
from tsl.nn.models.imputation import GRINModel
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values
from tsl.utils import parser_utils, numpy_metrics
from tsl.utils.parser_utils import ArgParser

from spin.baselines import SAITS, TransformerModel, BRITS, LSTMModel
from spin.imputers import SPINImputer, SAITSImputer, BRITSImputer, LSTMImputer
from spin.models import SPINModel, SPINHierarchicalModel
from spin.scheduler import CosineSchedulerWithRestarts
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
from spin.datasets.mask_switching_callback import MaskSwitchingCallback
from spin.datasets.bounded_imputation_dataset import filter_cross_boundary_windows


def get_model_classes(model_str):
    if model_str == 'spin':
        model, filler = SPINModel, SPINImputer
    elif model_str == 'spin_h':
        model, filler = SPINHierarchicalModel, SPINImputer
    elif model_str == 'grin':
        model, filler = GRINModel, Imputer
    elif model_str == 'saits':
        model, filler = SAITS, SAITSImputer
    elif model_str == 'transformer':
        model, filler = TransformerModel, SPINImputer
    elif model_str == 'brits':
        model, filler = BRITS, BRITSImputer
    elif model_str == 'lstm':
        model, filler = LSTMModel, LSTMImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name: str, data_path: str = None, 
                static_data_path: str = None, mask_data_path: str = None,
                feature_cols: list = None, data_groups: list = None,
                mask_files: list = None):
    """
    获取数据集
    
    Args:
        dataset_name: 数据集名称
        data_path: 动态交通数据路径 (用于lane数据集，单组模式)
        static_data_path: 静态道路数据路径 (用于lane数据集，单组模式)
        mask_data_path: 掩码文件路径 (用于lane数据集，单组模式)
        feature_cols: 特征列名列表 (用于lane数据集)
        data_groups: 多组数据配置列表 (用于lane数据集，多组模式)
        mask_files: 训练时随机选择的mask文件列表 (用于lane数据集)
    """
    # 支持车道级交通数据集
    if dataset_name == 'lane':
        if data_groups is not None:
            # 多组数据模式
            return LaneTrafficDataset(
                data_groups=data_groups,
                mask_files=mask_files,
                feature_cols=feature_cols,
                impute_nans=True
            )
        elif static_data_path is not None and data_path is not None:
            # 单组数据模式（向后兼容）
            return LaneTrafficDataset(
                static_data_path=static_data_path,
                dynamic_data_path=data_path,
                mask_data_path=mask_data_path,
                mask_files=mask_files,
                feature_cols=feature_cols,
                impute_nans=True
            )
        else:
            raise ValueError("lane数据集需要指定 data_groups 或 (--static-data-path + --data-path)")
    
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    # build missing dataset
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    raise ValueError(f"Invalid dataset name: {dataset_name}.")


def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    elif scheduler_name == 'magic':
        scheduler_class = CosineSchedulerWithRestarts
        scheduler_kwargs = dict(num_warmup_steps=12, min_factor=0.1,
                                linear_decay=0.67,
                                num_training_steps=args.epochs,
                                num_cycles=args.epochs // 100)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs

def check_shared_storage(model):
    storage_to_names = {}
    shared_found = False

    # 检查所有 buffer 和 parameter
    for name, tensor in model.named_parameters():
        storage_ptr = tensor.untyped_storage().data_ptr()  # 使用 untyped_storage
        if storage_ptr in storage_to_names:
            if not shared_found:
                print(f"💥 SHARED STORAGE DETECTED (Parameter or Buffer)!")
                shared_found = True
            print(f"   {storage_to_names[storage_ptr]} 和 {name} 共享同一块内存:")
            print(f"   Storage ptr: {storage_ptr}, Shape: {tensor.shape}")
        else:
            storage_to_names[storage_ptr] = name

    for name, buf in model.named_buffers():
        storage_ptr = buf.untyped_storage().data_ptr()
        if storage_ptr in storage_to_names:
            if not shared_found:
                print(f"💥 SHARED STORAGE DETECTED (Parameter or Buffer)!")
                shared_found = True
            print(f"   {storage_to_names[storage_ptr]} 和 {name} 共享同一块内存:")
            print(f"   Storage ptr: {storage_ptr}, Shape: {buf.shape}")
        else:
            storage_to_names[storage_ptr] = name

    if not shared_found:
        print("✅ 所有 Parameters 和 Buffers 内存独立，无共享")


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument("--model-name", type=str, default='spin')
    parser.add_argument("--dataset-name", type=str, default='air36')
    parser.add_argument("--data-path", type=str, default=None, 
                       help="Path to dynamic traffic data file (csv)")
    parser.add_argument("--static-data-path", type=str, default=None,
                       help="Path to static road data file (graph.json)")
    parser.add_argument("--mask-data-path", type=str, default=None,
                       help="Path to mask data file (csv)")
    parser.add_argument("--feature-cols", type=str, default=None,
                       help="Comma-separated feature column names")
    parser.add_argument("--config", type=str, default='imputation/spin.yaml')

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--min-delta', type=float, default=0.0,
                       help='Early stopping minimum delta. Validation metric must improve by at least this amount.')
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    
    # Checkpoint params
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help="Path to checkpoint file. If provided, skip training and load from checkpoint for testing.")
    parser.add_argument('--skip-train', action='store_true',
                       help="Skip training and only do testing (requires --checkpoint-path)")

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, imputer_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = imputer_cls.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    
    # 保存checkpoint相关参数，防止被yaml覆盖
    checkpoint_path = args.checkpoint_path
    skip_train = args.skip_train
    
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])
    
    # 恢复checkpoint相关参数
    args.checkpoint_path = checkpoint_path
    args.skip_train = skip_train
    
    # 确保数值参数被正确转换为浮点数（防止YAML解析为字符串）
    if hasattr(args, 'l2_reg'):
        args.l2_reg = float(args.l2_reg)
    if hasattr(args, 'min_delta'):
        args.min_delta = float(args.min_delta)
    if hasattr(args, 'lr'):
        args.lr = float(args.lr)
    
    # 处理 dataset_name 可能是列表的情况（从YAML配置中加载时）
    if isinstance(args.dataset_name, list):
        args.dataset_name = args.dataset_name[0]

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)
    
    # 内存优化设置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # 设置Tensor Cores精度以提升性能
        torch.set_float32_matmul_precision('medium')
    
    # 设置环境变量以避免内存共享问题
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # script flags
    is_spin = args.model_name in ['spin', 'spin_h']
    is_lstm = args.model_name == 'lstm'

    model_cls, imputer_class = get_model_classes(args.model_name)
    
    # 解析特征列
    feature_cols = None
    if args.feature_cols:
        feature_cols = [col.strip() for col in args.feature_cols.split(',')]
    
    # 获取 data_groups 配置（如果有）
    data_groups = getattr(args, 'data_groups', None)
    # 获取 mask_files 配置（如果有）
    mask_files = getattr(args, 'mask_files', None)
    
    dataset = get_dataset(
        args.dataset_name, 
        args.data_path,
        static_data_path=args.static_data_path,
        mask_data_path=args.mask_data_path,
        feature_cols=feature_cols,
        data_groups=data_groups,
        mask_files=mask_files
    )

    logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # time embedding
    if is_spin or args.model_name == 'transformer':
        time_emb = dataset.datetime_encoded([]).values
        exog_map = {'global_temporal_encoding': time_emb}

        input_map = {
            'u': 'temporal_encoding',
            'x': 'data'
        }
    else:
        exog_map = input_map = None

    if is_spin or args.model_name == 'grin':
        adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                       include_self=False,
                                       force_symmetric=is_spin)
        # 将邻接矩阵转换为 edge_index 格式 (2, num_edges)
        from tsl.ops.connectivity import adj_to_edge_index
        edge_index, edge_weight = adj_to_edge_index(adj)
        connectivity = (edge_index, edge_weight)
    elif is_lstm:
        # LSTM不需要图结构，但为了兼容性，可以设置为None
        connectivity = None
    else:
        connectivity = None

    # instantiate dataset
    data, index, node_ids = dataset.numpy(return_idx=True)
    torch_dataset = ImputationDataset(data=data,
                                      index=index,
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=connectivity,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)
    
    # 如果数据集有文件边界信息，过滤跨越边界的窗口
    if hasattr(dataset, 'file_boundaries') and dataset.file_boundaries:
        print(f"\n🔍 检测到 {len(dataset.file_boundaries)} 个文件边界，开始过滤跨越边界的窗口...")
        torch_dataset = filter_cross_boundary_windows(
            torch_dataset, 
            dataset.file_boundaries, 
            args.window
        )

    # get train/val/test indices
    splitter = dataset.get_splitter(args.val_len, args.test_len)

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size // args.split_batch_in)
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    input_size=dm.n_channels,
                                    u_size=2,
                                    output_size=dm.n_channels,
                                    window_size=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class,
                                                       return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        **imputer_kwargs
    )

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=args.patience, 
                                        mode='min',
                                        min_delta=getattr(args, 'min_delta', 0.0),
                                        check_on_train_epoch_end=False)
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1,
                                          monitor='val_mae', mode='min')

    tb_logger = TensorBoardLogger(logdir, name="model")
    
    # 如果指定了mask_files，添加mask切换回调
    callbacks = [early_stop_callback, checkpoint_callback]
    if mask_files and len(mask_files) > 0:
        mask_switching_callback = MaskSwitchingCallback(dataset, torch_dataset)
        callbacks.append(mask_switching_callback)
        print(f"✅ 已启用mask动态切换功能，共 {len(mask_files)} 个mask文件")
    else:
        print("ℹ️  未指定mask_files，使用固定的mask模式")
    
    # 确定checkpoint路径
    if args.checkpoint_path is not None:
        # 使用用户指定的checkpoint
        best_model_path = args.checkpoint_path
        print(f"使用指定的checkpoint: {best_model_path}")
    elif args.skip_train:
        raise ValueError("--skip-train 需要指定 --checkpoint-path 参数")
    else:
        best_model_path = None
    
    # 只在需要训练时创建trainer并训练
    if not args.skip_train:
        print("开始训练...")
        print("Checking shared storage...here!!!!!!!")
        
        trainer = pl.Trainer(max_epochs=args.epochs,
                             default_root_dir=logdir,
                             logger=tb_logger,
                             precision=args.precision,
                             accumulate_grad_batches=args.split_batch_in,
                             accelerator='gpu', 
                             devices=1,
                             gradient_clip_val=args.grad_clip_val,
                             limit_train_batches=args.batches_epoch * args.split_batch_in,
                             check_val_every_n_epoch=1,
                             log_every_n_steps=1,
                             callbacks=callbacks)
        check_shared_storage(imputer)
        print("Checking shared storage...done!!!!!!!")
        trainer.fit(imputer,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader(
                        batch_size=args.batch_inference))
        
        # 训练完成后使用最佳模型
        best_model_path = checkpoint_callback.best_model_path
        print(f"训练完成，最佳模型: {best_model_path}")
    else:
        print("跳过训练，直接使用checkpoint进行测试...")
    
    ########################################
    # testing                              #
    ########################################

    # 创建测试用的trainer
    test_trainer = pl.Trainer(accelerator='gpu', devices=1, precision=args.precision)
    
    # 从checkpoint加载模型权重
    if best_model_path is not None:
        print(f"开始测试，从checkpoint加载模型: {best_model_path}")
        # 使用 load_from_checkpoint 加载模型
        imputer = imputer_class.load_from_checkpoint(
            best_model_path,
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': args.lr, 'weight_decay': args.l2_reg},
            loss_fn=loss_fn,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            **imputer_kwargs
        )
    else:
        print("使用当前训练好的模型进行测试")
    
    # 测试
    test_trainer.test(imputer, dataloaders=dm.test_dataloader(
        batch_size=args.batch_inference))

    # 预测
    output_list = test_trainer.predict(imputer, dataloaders=dm.test_dataloader(
        batch_size=args.batch_inference))
    
    # 将字典列表合并为单个字典，每个键包含所有批次的拼接结果
    # output_list 是一个字典列表，每个字典包含 'y_hat', 'y', 'mask'
    y_hat_list = []
    y_list = []
    mask_list = []
    
    for batch_output in output_list:
        y_hat_list.append(batch_output['y_hat'].detach().cpu())
        y_list.append(batch_output['y'].detach().cpu())
        mask_list.append(batch_output['mask'].detach().cpu())
    
    # 拼接所有批次
    y_hat = torch.cat(y_hat_list, dim=0).numpy()
    y_true = torch.cat(y_list, dim=0).numpy()
    mask = torch.cat(mask_list, dim=0).numpy()
    
    # 只在最后一个维度大小为1时才squeeze
    if y_hat.shape[-1] == 1:
        y_hat = y_hat.squeeze(-1)
    if y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    
    check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    print(f'Test MAE: {check_mae:.2f}')
    return y_hat


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
