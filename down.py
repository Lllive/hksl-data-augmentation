from huggingface_hub import snapshot_download

# 这里改成了图一里的正确模型ID
model_id = "Qwen/Qwen2.5-7B-Instruct"

# 下载到当前目录下的 models 文件夹里
snapshot_download(
    repo_id=model_id,
    local_dir="models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,  # 确保下载的是实体文件
    resume_download=True           # 支持断点续传
)

print("下载完成！")

