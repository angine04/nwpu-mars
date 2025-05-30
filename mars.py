import sys
import argparse
from engine.engine import MarsEngine


def parse_legacy_args():
    """解析传统的命令行参数格式，保持向后兼容"""
    mode = "pipe"
    nobuf = False
    cfgname = None
    root = None
    
    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"
    
    return mode, nobuf, cfgname, root


def parse_enhanced_args():
    """解析增强的命令行参数"""
    parser = argparse.ArgumentParser(
        description='MARS YOLOv8 Training and Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 传统用法
  python mars.py train vanilla.nano.fast.pretrained
  python mars.py -train
  python mars.py -eval -nobuf
  
  # 新的灵活用法
  python mars.py train --config vanilla.nano.fast.pretrained.ema.clsboost
  python mars.py train --config vanilla.nano.full --tags pretrained,ema,earlystop,focalweight
  python mars.py eval --config vanilla.nano.fast --root /custom/path --nobuf
  python mars.py train --config vanilla.nano.fast --batch-size 32 --max-epochs 100
  python mars.py train --config vanilla.nano.fast --device cuda:1 --learning-rate 0.02
  
  # 快速配置
  python mars.py train --quick-config full-optimized  # 等同于 vanilla.nano.full.pretrained.ema.earlystop.focalweight
  python mars.py train --quick-config fast-test       # 等同于 vanilla.nano.fast.pretrained.ema
  python mars.py train --quick-config distillation    # 等同于 vanilla.nano.distillation.ema
        """
    )
    
    # 主要命令
    parser.add_argument('command', nargs='?', choices=['train', 'eval', 'pipe'], 
                       help='运行模式：train(训练), eval(评估), pipe(训练+评估)')
    
    # 位置参数（用于向后兼容）
    parser.add_argument('config_name', nargs='?', 
                       help='配置名称（如：vanilla.nano.fast.pretrained）')
    
    # 配置相关参数
    parser.add_argument('--config', '-c', type=str,
                       help='配置名称（如：vanilla.nano.fast.pretrained）')
    parser.add_argument('--tags', '-t', type=str,
                       help='额外的配置标签，逗号分隔（如：pretrained,ema,clsboost）')
    parser.add_argument('--quick-config', '-q', type=str,
                       choices=['full-optimized', 'fast-test', 'distillation', 'debug'],
                       help='快速配置预设')
    
    # 系统参数
    parser.add_argument('--root', '-r', type=str, default='/home/v5/Mars',
                       help='项目根目录路径（默认：/home/v5/Mars）')
    parser.add_argument('--nobuf', action='store_true',
                       help='禁用缓存，强制重新训练')
    
    # 训练参数覆盖
    parser.add_argument('--batch-size', '-b', type=int,
                       help='批次大小（覆盖配置文件设置）')
    parser.add_argument('--max-epochs', '-e', type=int,
                       help='最大训练轮数（覆盖配置文件设置）')
    parser.add_argument('--learning-rate', '-lr', type=float,
                       help='学习率（覆盖配置文件设置）')
    parser.add_argument('--device', '-d', type=str,
                       help='设备（如：cuda:0, cuda:1, cpu, mps）')
    parser.add_argument('--phase', '-p', type=str,
                       help='模型阶段（如：nano, small, medium）')
    
    # 损失权重参数
    parser.add_argument('--cls-weight', type=float,
                       help='分类损失权重（覆盖配置文件设置）')
    parser.add_argument('--box-weight', type=float,
                       help='边界框损失权重（覆盖配置文件设置）')
    parser.add_argument('--dfl-weight', type=float,
                       help='DFL损失权重（覆盖配置文件设置）')
    
    # Early stopping参数
    parser.add_argument('--early-stop-patience', type=int,
                       help='Early stopping耐心值（覆盖配置文件设置）')
    
    # 调试和信息参数
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出模式')
    parser.add_argument('--dry-run', action='store_true',
                       help='干运行模式，只显示配置不执行训练')
    
    return parser.parse_args()


def resolve_quick_config(quick_config):
    """解析快速配置预设"""
    quick_configs = {
        'full-optimized': 'vanilla.nano.full.pretrained.ema.earlystop.focalweight',
        'fast-test': 'vanilla.nano.fast.pretrained.ema',
        'distillation': 'vanilla.nano.distillation.ema',
        'debug': 'vanilla.nano.fast'
    }
    return quick_configs.get(quick_config, quick_config)


def build_config_name(args):
    """构建最终的配置名称"""
    config_name = None
    
    # 优先级：--config > --quick-config > 位置参数 > 默认值
    if args.config:
        config_name = args.config
    elif args.quick_config:
        config_name = resolve_quick_config(args.quick_config)
    elif args.config_name:
        config_name = args.config_name
    else:
        config_name = "vanilla.nano.fast.pretrained.focalweight"
    
    # 添加额外的tags
    if args.tags:
        additional_tags = args.tags.split(',')
        config_name = config_name + '.' + '.'.join(additional_tags)
    
    return config_name


def apply_parameter_overrides(mcfg, args):
    """应用命令行参数覆盖"""
    if args.batch_size:
        mcfg.batchSize = args.batch_size
        if args.verbose:
            print(f"覆盖批次大小: {args.batch_size}")
    
    if args.max_epochs:
        mcfg.maxEpoch = args.max_epochs
        if args.verbose:
            print(f"覆盖最大轮数: {args.max_epochs}")
    
    if args.learning_rate:
        mcfg.baseLearningRate = args.learning_rate
        if args.verbose:
            print(f"覆盖学习率: {args.learning_rate}")
    
    if args.device:
        import torch
        mcfg.device = torch.device(args.device)
        if args.verbose:
            print(f"覆盖设备: {args.device}")
    
    if args.phase:
        mcfg.phase = args.phase
        if args.verbose:
            print(f"覆盖模型阶段: {args.phase}")
    
    # 损失权重覆盖
    if any([args.cls_weight, args.box_weight, args.dfl_weight]):
        current_weights = list(mcfg.lossWeights)
        if args.box_weight:
            current_weights[0] = args.box_weight
        if args.cls_weight:
            current_weights[1] = args.cls_weight
        if args.dfl_weight:
            current_weights[2] = args.dfl_weight
        mcfg.lossWeights = tuple(current_weights)
        if args.verbose:
            print(f"覆盖损失权重: {mcfg.lossWeights}")
    
    if args.early_stop_patience:
        mcfg.early_stopping_patience = args.early_stop_patience
        if args.verbose:
            print(f"覆盖Early stopping耐心值: {args.early_stop_patience}")
    
    return mcfg


def print_config_summary(args, config_name, mcfg):
    """打印配置摘要"""
    print("=" * 60)
    print("MARS YOLOv8 配置摘要")
    print("=" * 60)
    print(f"运行模式: {args.command}")
    print(f"配置名称: {config_name}")
    print(f"项目根目录: {args.root}")
    print(f"禁用缓存: {args.nobuf}")
    print(f"设备: {mcfg.device}")
    print(f"批次大小: {mcfg.batchSize}")
    print(f"最大轮数: {mcfg.maxEpoch}")
    print(f"学习率: {mcfg.baseLearningRate}")
    print(f"损失权重: {mcfg.lossWeights}")
    if hasattr(mcfg, 'use_ema') and mcfg.use_ema:
        print(f"EMA启用: True")
    if hasattr(mcfg, 'use_early_stopping') and mcfg.use_early_stopping:
        print(f"Early Stopping启用: True (耐心值: {mcfg.early_stopping_patience})")
    if hasattr(mcfg, 'use_focal_cls_weight') and mcfg.use_focal_cls_weight:
        print(f"Focal Loss权重调整启用: True")
    print("=" * 60)


if __name__ == "__main__":
    # 检测是否使用传统参数格式
    is_legacy_format = any(arg.startswith('-') and arg in ['-train', '-eval', '-pipe', '-nobuf'] 
                          for arg in sys.argv[1:])
    
    if is_legacy_format or (len(sys.argv) >= 2 and sys.argv[1] in ['train', 'eval', 'pipe'] and len(sys.argv) <= 3):
        # 传统格式兼容模式
        if len(sys.argv) >= 3 and not sys.argv[2].startswith('-'):
            # python mars.py train vanilla.nano.fast.pretrained
            mode = sys.argv[1]
            config_name = sys.argv[2]
            nobuf = '-nobuf' in sys.argv
        else:
            # python mars.py -train -nobuf
            mode, nobuf, _, _ = parse_legacy_args()
            config_name = "vanilla.nano.fast.pretrained"
        
        root = "/home/v5/Mars"
        
        print(f"使用传统兼容模式: {mode} {config_name}")
        
    else:
        # 新的增强参数格式
        args = parse_enhanced_args()
        
        # 处理命令和配置
        mode = args.command or "pipe"
        config_name = build_config_name(args)
        root = args.root
        nobuf = args.nobuf
        
        if args.verbose:
            print(f"使用增强参数模式")
            print(f"解析的配置名称: {config_name}")
    
    # 创建引擎
    engine = MarsEngine(
        mode=mode,
        cfgname=config_name,
        root=root,
        nobuf=nobuf,
    )
    
    # 如果使用增强模式，应用参数覆盖
    if not is_legacy_format and not (len(sys.argv) >= 2 and sys.argv[1] in ['train', 'eval', 'pipe'] and len(sys.argv) <= 3):
        args = parse_enhanced_args()
        engine.mcfg = apply_parameter_overrides(engine.mcfg, args)
        
        if args.verbose or args.dry_run:
            print_config_summary(args, config_name, engine.mcfg)
        
        if args.dry_run:
            print("干运行模式，退出...")
            sys.exit(0)
    
    # 运行引擎
    engine.run()
