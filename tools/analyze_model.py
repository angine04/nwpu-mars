#!/usr/bin/env python3
"""
分析官方YOLOv8预训练模型文件结构并对比项目模型
"""
import torch
import pprint
import sys
import os

# 添加项目路径
sys.path.append('.')

def analyze_official_model():
    print("=== 分析官方YOLOv8n模型文件 ===")
    
    # 加载官方YOLOv8n预训练模型
    model_dict = torch.load('yolov8n_official.pt', map_location='cpu', weights_only=False)
    
    print('\n=== 模型文件顶层结构 ===')
    for key in model_dict.keys():
        print(f'{key}: {type(model_dict[key])}')
    
    # 提取模型部分
    if 'model' in model_dict:
        model = model_dict['model']
        print(f'\n=== 模型对象类型 ===')
        print(f'模型类型: {type(model)}')
        
        # 获取state_dict
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model
        
        print(f'\n=== 模型参数总数 ===')
        print(f'参数总数: {len(state_dict)}')
        
        print(f'\n=== 模型参数名称和形状 (前30个) ===')
        for i, (name, tensor) in enumerate(state_dict.items()):
            print(f'{i:2d}: {name:60s} {str(tensor.shape):20s}')
            if i >= 29:  # 显示前30个参数
                remaining = len(state_dict) - 30
                if remaining > 0:
                    print(f'    ... (还有{remaining}个参数)')
                break
        
        # 分析backbone相关参数
        print(f'\n=== Backbone相关参数 (层0-9) ===')
        backbone_params = {k: v for k, v in state_dict.items() if 'model.' in k and any(x in k for x in ['0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'])}
        for name, tensor in list(backbone_params.items())[:20]:
            print(f'{name:60s} {str(tensor.shape):20s}')
        
        if len(backbone_params) > 20:
            print(f'    ... (backbone还有{len(backbone_params)-20}个参数)')
            
        return state_dict
    
    return None

def analyze_project_model():
    print("\n\n=== 分析项目模型结构 ===")
    
    try:
        # 导入项目配置和模型
        from config.mconfig import ModelConfig
        from model.base.yolomodel import YoloModel
        
        # 创建模型配置
        mcfg = ModelConfig()
        mcfg.phase = "nano"
        mcfg.nc = 20  # MAR20数据集的类别数
        mcfg.regMax = 16
        mcfg.inputShape = (640, 640)
        mcfg.device = torch.device('cpu')
        
        # 创建项目模型
        project_model = YoloModel(mcfg)
        project_state_dict = project_model.state_dict()
        
        print(f'\n=== 项目模型参数总数 ===')
        print(f'参数总数: {len(project_state_dict)}')
        
        print(f'\n=== 项目模型参数名称和形状 (前30个) ===')
        for i, (name, tensor) in enumerate(project_state_dict.items()):
            print(f'{i:2d}: {name:60s} {str(tensor.shape):20s}')
            if i >= 29:  # 显示前30个参数
                remaining = len(project_state_dict) - 30
                if remaining > 0:
                    print(f'    ... (还有{remaining}个参数)')
                break
        
        # 分析backbone相关参数
        print(f'\n=== 项目Backbone相关参数 ===')
        backbone_params = {k: v for k, v in project_state_dict.items() if 'backbone.' in k}
        for name, tensor in list(backbone_params.items())[:20]:
            print(f'{name:60s} {str(tensor.shape):20s}')
        
        if len(backbone_params) > 20:
            print(f'    ... (backbone还有{len(backbone_params)-20}个参数)')
            
        return project_state_dict
    
    except Exception as e:
        print(f'分析项目模型时出错: {e}')
        import traceback
        traceback.print_exc()
        return None

def compare_models(official_state_dict, project_state_dict):
    if official_state_dict is None or project_state_dict is None:
        print("无法进行模型对比，因为有模型加载失败")
        return
    
    print("\n\n=== 模型参数对比分析 ===")
    
    # 分析官方模型的backbone部分 (层0-9)
    official_backbone = {k: v for k, v in official_state_dict.items() 
                        if k.startswith('model.') and any(f'model.{i}.' in k for i in range(10))}
    
    # 分析项目模型的backbone部分
    project_backbone = {k: v for k, v in project_state_dict.items() if k.startswith('backbone.')}
    
    print(f'官方模型backbone参数数: {len(official_backbone)}')
    print(f'项目模型backbone参数数: {len(project_backbone)}')
    
    print(f'\n=== 参数名称映射分析 ===')
    print('官方模型 -> 项目模型映射关系:')
    
    official_keys = list(official_backbone.keys())[:10]
    project_keys = list(project_backbone.keys())[:10]
    
    for i, (off_key, proj_key) in enumerate(zip(official_keys, project_keys)):
        off_shape = official_backbone[off_key].shape
        proj_shape = project_backbone[proj_key].shape
        match = "✓" if off_shape == proj_shape else "✗"
        print(f'{i:2d}: {off_key:50s} -> {proj_key:50s} {match}')
        print(f'    {str(off_shape):20s} -> {str(proj_shape):20s}')

if __name__ == '__main__':
    official_state_dict = analyze_official_model()
    project_state_dict = analyze_project_model()
    compare_models(official_state_dict, project_state_dict) 