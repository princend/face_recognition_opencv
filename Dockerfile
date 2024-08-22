# 使用Python的官方镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 将requirements.txt文件复制到工作目录
COPY requirements.txt .

# 安装系统依赖工具
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0    

# 安装Python依赖包
RUN pip install --no-cache-dir -r requirements.txt


# 将应用程序代码复制到工作目录
COPY . .

# 对外暴露端口
EXPOSE 8080

# 运行Flask应用
# CMD ["gunicorn", "--bind", "0.0.0.0:5000","--timeout", "200", "app:app"]
CMD ["python", "app.py"]

