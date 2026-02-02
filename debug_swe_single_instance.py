#!/usr/bin/env python3
"""
单个 R2E-Gym Docker 环境测试脚本（使用 Kubernetes 后端）

这个脚本用于测试本地 k8s 配置是否正确，能否成功运行单个 SWE instance。
运行前请确保:
1. 本地已配置好 k8s config (kubectl get nodes 能看到集群)
2. 已安装 R2E-Gym: pip install -e git+https://github.com/R2E-Gym/R2E-Gym.git
3. 已安装 rllm: pip install -e .
"""

import asyncio
import os
import sys
from datasets import load_dataset
from rllm.environments.swe.swe import SWEEnv


def test_single_instance_sync(dataset_name="R2E-Gym/R2E-Gym-Lite", instance_idx=0):
    """
    同步测试单个 instance，使用 kubernetes 后端
    
    Args:
        dataset_name: R2E-Gym 数据集名称
        instance_idx: 要测试的 instance 索引
    """
    print("=" * 80)
    print("🧪 开始测试单个 R2E-Gym Instance (Kubernetes 后端)")
    print("=" * 80)
    
    # 1. 加载数据集
    print(f"\n📊 加载数据集: {dataset_name}")
    try:
        ds = load_dataset(dataset_name, split="train")
        print(f"✅ 数据集加载成功，共 {len(ds)} 个 instances")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False
    
    # 检查索引是否有效
    if instance_idx >= len(ds):
        print(f"❌ 索引 {instance_idx} 超出范围 (0-{len(ds)-1})")
        return False
    
    entry = ds[instance_idx]
    print(f"\n📝 Instance ID: {entry.get('instance_id', 'N/A')}")
    print(f"📝 Repo: {entry.get('repo', 'N/A')}")
    print(f"📝 Original Docker Image: {entry.get('docker_image', 'N/A')}")
    
    # 显示将要使用的 Docker 镜像前缀
    mirror_prefix = os.environ.get("DOCKER_MIRROR_PREFIX", "")
    if mirror_prefix:
        original_image = entry.get('docker_image', 'N/A')
        if original_image != 'N/A' and '/' in original_image:
            first_part = original_image.split('/')[0]
            if '.' not in first_part:
                expected_image = f"{mirror_prefix}/{original_image}"
                print(f"📝 Expected Docker Image (with mirror): {expected_image}")
    
    # 2. 创建 SWE 环境（使用 kubernetes 后端）
    print(f"\n🚀 创建 SWE 环境 (backend=kubernetes)...")
    try:
        env = SWEEnv(
            entry=entry,
            backend='kubernetes',  # 使用 kubernetes 后端
            scaffold='r2egym',     # 使用 r2egym scaffold
            step_timeout=120,      # 步骤超时 120 秒
            reward_timeout=300,    # 奖励计算超时 300 秒
            delete_image=False,    # 测试时不删除镜像
            verbose=True,          # 详细输出
        )
        print("✅ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 重置环境（这会启动 Docker 容器）
    print("\n🔄 重置环境（启动 Kubernetes Pod）...")
    try:
        task_instruction, info = env.reset()
        print("✅ 环境重置成功")
        print(f"\n📋 任务描述:\n{task_instruction[:500]}...")
    except Exception as e:
        print(f"❌ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False
    
    # 4. 执行一个简单的测试步骤
    print("\n🧪 执行测试步骤...")
    test_actions = [
        "execute_bash pwd",  # 测试 bash 命令
        "search_dir .",      # 测试搜索功能
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n步骤 {i+1}: {action}")
        try:
            obs, reward, done, info = env.step(action)
            print(f"✅ 步骤执行成功")
            print(f"   观察结果: {obs[:200]}...")
            print(f"   奖励: {reward}")
            print(f"   完成: {done}")
            
            if done:
                print("✅ 任务已完成")
                break
        except Exception as e:
            print(f"❌ 步骤执行失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 5. 清理环境
    print("\n🧹 清理环境...")
    try:
        env.close()
        print("✅ 环境清理成功")
    except Exception as e:
        print(f"⚠️ 环境清理时出现警告: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    return True


def check_k8s_config():
    """检查 k8s 配置是否正确"""
    print("🔍 检查 Kubernetes 配置...")
    
    # 检查 kubectl 是否可用
    ret = os.system("kubectl version --client > /dev/null 2>&1")
    if ret != 0:
        print("❌ kubectl 未安装或不在 PATH 中")
        print("   请安装 kubectl: https://kubernetes.io/docs/tasks/tools/")
        return False
    print("✅ kubectl 已安装")
    
    # 检查是否能连接到集群
    ret = os.system("kubectl get nodes > /dev/null 2>&1")
    if ret != 0:
        print("❌ 无法连接到 Kubernetes 集群")
        print("   请检查 ~/.kube/config 配置")
        return False
    
    # 显示集群信息
    print("✅ Kubernetes 集群连接正常")
    print("\n集群节点:")
    os.system("kubectl get nodes")
    
    return True


def main():
    """主函数"""
    print("R2E-Gym Kubernetes 后端单实例测试工具\n")
    
    # 1. 检查 k8s 配置
    if not check_k8s_config():
        print("\n❌ Kubernetes 配置检查失败，请先配置好 k8s")
        sys.exit(1)
    
    # 2. 检查依赖
    print("\n🔍 检查 Python 依赖...")
    try:
        import r2egym
        print("✅ R2E-Gym 已安装")
    except ImportError:
        print("❌ R2E-Gym 未安装")
        print("   请运行: git clone https://github.com/R2E-Gym/R2E-Gym.git && cd R2E-Gym && pip install -e .")
        sys.exit(1)
    
    try:
        import rllm
        print("✅ rLLM 已安装")
    except ImportError:
        print("❌ rLLM 未安装")
        print("   请运行: pip install -e .")
        sys.exit(1)
    
    # 3. 运行测试
    print("\n" + "=" * 80)
    dataset_name = os.getenv("DATASET_NAME", "R2E-Gym/R2E-Gym-Subset")
    instance_idx = int(os.getenv("INSTANCE_IDX", "0"))
        
    success = test_single_instance_sync(
        dataset_name=dataset_name,
        instance_idx=instance_idx
    )
    
    if not success:
        print("\n❌ 测试失败")
        sys.exit(1)
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    # 设置环境变量（可选）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # 设置 Docker 镜像前缀为中国区镜像
    os.environ.setdefault("DOCKER_MIRROR_PREFIX", "aibrix-docker-mirror-cn-beijing.cr.volces.com")
    
    # 运行主函数
    main()
