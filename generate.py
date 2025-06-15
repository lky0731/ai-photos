# generate_interactive.py
import torch
from diffusers import DiffusionPipeline
import os
import warnings
import time

# 禁用可能产生问题的组件
warnings.filterwarnings("ignore", category=UserWarning)

# 设置模型路径
MODEL_PATH = "./moldle/LCM_Dreamshaper_v7"


def check_prerequisites():
    """检查环境和模型是否准备就绪"""
    # 确保模型路径存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型路径不存在 {os.path.abspath(MODEL_PATH)}")
        print("请检查: 1) moldle/LCM_Dreamshaper_v7 目录是否存在")
        print("        2) 模型文件是否下载完整")
        return False

    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("❌ 错误: CUDA不可用! 请确保安装了GPU版本的PyTorch")
        print("解决方案: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    return True


def load_model():
    """加载AI模型"""
    print("⏳ 正在加载AI模型...")
    start_time = time.time()

    try:
        # 加载模型
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None  # 禁用安全检查器
        )
        pipe.to("cuda")

        # 可选优化
        try:
            pipe.enable_attention_slicing()  # 减少显存占用
            print("  已启用注意力切片优化 - 减少显存占用")
        except:
            print("  注意力切片优化不可用")

        # 打印模型信息
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功! 耗时: {load_time:.2f}秒")
        return pipe
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print("请尝试: pip install --force-reinstall diffusers")
        return None


def prompt_builder():
    """构建用户交互提示"""
    print("\n🎨 请输入图像描述 (例如: '1girl, anime style')")
    prompt = input("> ")

    # 可选添加默认修饰词
    if "best quality" not in prompt.lower():
        prompt += ", best quality"
    if "masterpiece" not in prompt.lower():
        prompt += ", masterpiece"

    return prompt


def get_generation_settings():
    """获取用户生成设置"""
    print("\n⚙️ 请选择生成参数 (默认按回车使用推荐设置)")

    # 分辨率选择
    res_options = {
        "1": {"h": 768, "w": 512, "desc": "高清 (768x512)"},
        "2": {"h": 512, "w": 384, "desc": "标准 (512x384) - 推荐", "default": True},
        "3": {"h": 384, "w": 256, "desc": "快速 (384x256)"}
    }

    print("\n请选择分辨率:")
    for key, val in res_options.items():
        print(f"  [{key}] {val['desc']} {'(默认)' if val.get('default', False) else ''}")

    res_choice = input("> ") or "2"
    settings = res_options.get(res_choice, res_options["2"])

    # 生成步数
    print("\n请设置生成步数 (默认:4, 范围1-8):")
    steps = input("> ") or "4"
    try:
        settings["steps"] = max(1, min(8, int(steps)))
    except:
        settings["steps"] = 4

    # 生成数量
    print("\n请输入生成数量 (默认:1):")
    count = input("> ") or "1"
    try:
        settings["count"] = max(1, min(4, int(count)))
    except:
        settings["count"] = 1

    # 引导比例 (CFG)
    print("\n请设置引导比例 (默认:0.8, 范围0-2):")
    guidance = input("> ") or "0.8"
    try:
        settings["guidance"] = max(0.0, min(2.0, float(guidance)))
    except:
        settings["guidance"] = 0.8

    return settings


def generate_images(pipe, prompt, settings):
    """使用用户设置生成图像"""
    print(f"\n🚀 开始生成: '{prompt}'")
    print(
        f"  设置: {settings['count']}张图像 | {settings['steps']}步 | {settings['h']}x{settings['w']} | 引导:{settings['guidance']}")

    start_time = time.time()

    try:
        # 显示生成进度
        print(f"  0% [{' ' * 30}]", end="", flush=True)

        # 生成图像
        images = pipe(
            prompt=prompt,
            num_inference_steps=settings["steps"],
            num_images_per_prompt=settings["count"],
            height=settings["h"],
            width=settings["w"],
            guidance_scale=settings["guidance"]
        ).images

        # 计算耗时
        gen_time = time.time() - start_time
        speed = gen_time / settings["steps"] / settings["count"] if settings["count"] > 0 else gen_time

        # 打印成功消息
        print(f"\r✅ 生成成功! 耗时: {gen_time:.2f}秒 | 平均步速: {speed:.2f}秒/步")

        # 保存结果
        saved_paths = []
        for i, img in enumerate(images):
            filename = f"generated_{int(time.time())}_{i}.jpg"
            img.save(filename)
            saved_paths.append(filename)

        # 打印结果路径
        print("\n💾 图像已保存为:")
        for path in saved_paths:
            print(f"  - {os.path.abspath(path)}")

        return saved_paths
    except torch.cuda.OutOfMemoryError:
        print("\n💥 显存不足! 请尝试:")
        print("  1. 降低分辨率 (选择快速模式)")
        print("  2. 减少生成数量")
        print("  3. 减少生成步数")
        return []
    except Exception as e:
        print(f"\n🔥 生成错误: {str(e)}")
        return []


def main():
    """主交互界面"""
    print("=" * 50)
    print("🌟 二次元角色生成系统 v1.0")
    print("=" * 50)

    # 检查环境
    if not check_prerequisites():
        return

    # 显示GPU信息
    print(f"使用设备: {torch.cuda.get_device_name(0)}")
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    # 加载模型
    pipe = load_model()
    if pipe is None:
        return

    # 主循环
    while True:
        try:
            # 获取用户输入
            prompt = prompt_builder()
            settings = get_generation_settings()

            # 生成图像
            generate_images(pipe, prompt, settings)

            # 继续生成？
            print("\n继续生成新图像吗? [y/n]")
            choice = input("> ").lower()
            if choice != "y":
                break

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break


if __name__ == "__main__":
    main()