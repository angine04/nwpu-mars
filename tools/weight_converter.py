#!/usr/bin/env python3
"""
YOLOv8官方预训练权重转换工具
将官方模型的参数名转换为项目模型兼容的格式
"""
import torch
import os
import sys
from typing import Dict, Any

# 添加项目路径
sys.path.append('.')

def create_name_mapping():
    """
    创建官方模型参数名到项目模型参数名的映射规则
    """
    def map_parameter_name(official_name: str) -> str:
        """
        将官方模型参数名转换为项目模型参数名
        
        官方格式: model.{layer_index}.{component}.{parameter}
        项目格式: layers.{layer_index}.{component}.{parameter}
        """
        if not official_name.startswith('model.'):
            return None  # 非backbone参数，暂不处理
        
        # 提取层索引
        parts = official_name.split('.')
        if len(parts) < 2:
            return None
            
        layer_index = parts[1]
        
        # 只转换backbone部分 (层0-9)
        try:
            layer_num = int(layer_index)
            if 0 <= layer_num <= 9:
                # 替换前缀: model. -> layers.
                project_name = official_name.replace('model.', 'layers.')
                return project_name
        except ValueError:
            pass
            
        return None
    
    return map_parameter_name

def convert_weights(official_model_path: str, output_path: str):
    """
    转换官方预训练权重为项目兼容格式
    
    Args:
        official_model_path: 官方.pt文件路径
        output_path: 输出的转换后权重文件路径
    """
    print(f"=== YOLOv8权重转换工具 ===")
    print(f"输入文件: {official_model_path}")
    print(f"输出文件: {output_path}")
    
    # 加载官方模型
    print("\n1. 加载官方预训练模型...")
    model_dict = torch.load(official_model_path, map_location='cpu', weights_only=False)
    
    if 'model' not in model_dict:
        raise ValueError("官方模型文件格式不正确，缺少'model'键")
    
    # 获取模型参数
    official_model = model_dict['model']
    if hasattr(official_model, 'state_dict'):
        official_state_dict = official_model.state_dict()
    else:
        official_state_dict = official_model
    
    print(f"原始模型参数数: {len(official_state_dict)}")
    
    # 创建参数名映射函数
    map_name = create_name_mapping()
    
    # 转换参数
    print("\n2. 转换参数名...")
    converted_state_dict = {}
    converted_count = 0
    skipped_count = 0
    
    for official_name, tensor in official_state_dict.items():
        project_name = map_name(official_name)
        
        if project_name:
            converted_state_dict[project_name] = tensor.clone()
            converted_count += 1
            if converted_count <= 10:  # 显示前10个转换示例
                print(f"  {official_name} -> {project_name}")
        else:
            skipped_count += 1
    
    if converted_count > 10:
        print(f"  ... (共转换了{converted_count}个backbone参数)")
    
    print(f"\n转换统计:")
    print(f"  成功转换: {converted_count} 个参数")
    print(f"  跳过参数: {skipped_count} 个参数 (非backbone)")
    
    # 保存转换后的权重
    print(f"\n3. 保存转换后的权重...")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为项目兼容格式
    save_dict = {
        'state_dict': converted_state_dict,
        'source': 'official_yolov8n_converted',
        'conversion_info': {
            'original_file': official_model_path,
            'converted_params': converted_count,
            'skipped_params': skipped_count,
            'total_original_params': len(official_state_dict)
        }
    }
    
    torch.save(save_dict, output_path)
    print(f"权重转换完成! 保存至: {output_path}")
    
    return converted_state_dict

def verify_conversion(converted_weights_path: str):
    """
    验证转换后的权重是否能被项目模型正确加载
    """
    print(f"\n=== 验证权重转换结果 ===")
    
    try:
        # 导入项目模型
        from config.mconfig import ModelConfig
        from model.base.yolomodel import YoloModel
        
        # 创建项目模型
        mcfg = ModelConfig()
        mcfg.phase = "nano"
        mcfg.nc = 20
        mcfg.regMax = 16
        mcfg.inputShape = (640, 640)
        mcfg.device = torch.device('cpu')
        
        project_model = YoloModel(mcfg)
        
        # 加载转换后的权重
        checkpoint = torch.load(converted_weights_path, map_location='cpu')
        converted_state_dict = checkpoint['state_dict']
        
        # 尝试加载权重到backbone
        backbone_state_dict = project_model.backbone.state_dict()
        
        # 检查参数匹配情况
        missing_keys = []
        unexpected_keys = []
        size_mismatch_keys = []
        
        for name, param in backbone_state_dict.items():
            if name in converted_state_dict:
                if param.shape != converted_state_dict[name].shape:
                    size_mismatch_keys.append(f"{name}: 期望{param.shape}, 实际{converted_state_dict[name].shape}")
            else:
                missing_keys.append(name)
        
        for name in converted_state_dict.keys():
            if name not in backbone_state_dict:
                unexpected_keys.append(name)
        
        # 输出验证结果
        print(f"验证结果:")
        print(f"  转换权重参数数: {len(converted_state_dict)}")
        print(f"  项目backbone参数数: {len(backbone_state_dict)}")
        print(f"  缺失参数: {len(missing_keys)}")
        print(f"  多余参数: {len(unexpected_keys)}")
        print(f"  形状不匹配: {len(size_mismatch_keys)}")
        
        if missing_keys:
            print(f"\n缺失的参数 (前5个):")
            for key in missing_keys[:5]:
                print(f"    {key}")
                
        if unexpected_keys:
            print(f"\n多余的参数 (前5个):")
            for key in unexpected_keys[:5]:
                print(f"    {key}")
                
        if size_mismatch_keys:
            print(f"\n形状不匹配的参数:")
            for key in size_mismatch_keys:
                print(f"    {key}")
        
        # 尝试实际加载
        if not missing_keys and not size_mismatch_keys:
            print(f"\n✓ 所有参数匹配! 尝试加载权重...")
            project_model.backbone.load_state_dict(converted_state_dict, strict=True)
            print(f"✓ 权重加载成功!")
            return True
        else:
            print(f"\n✗ 存在参数不匹配，需要检查转换逻辑")
            return False
            
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 配置文件路径
    official_model_path = "yolov8n_official.pt"
    output_path = "yolov8n_converted_backbone.pth"
    
    if not os.path.exists(official_model_path):
        print(f"错误: 找不到官方模型文件 {official_model_path}")
        return
    
    try:
        # 执行权重转换
        converted_state_dict = convert_weights(official_model_path, output_path)
        
        # 验证转换结果
        verify_conversion(output_path)
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 