# hippo

#### 介绍
高级软件工程大作业：具有长记忆的问答智能体系统——hippo

#### 后端启动

1.  在命令行中输入以下命令(若无docker需先安装docker）：
```bash
docker run -d --name hippo-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=20031109@WJX -e POSTGRES_DB=hippo -p 5432:5432 ankane/pgvector
```
2.  进入hippo文件夹，在conda命令行中执行以下操作：
```bash
# 创建一个指定版本为 3.10 的新环境 (环境名可以叫 hippo_env)
conda create -n hippo_env python=3.10

# 激活新环境
conda activate hippo_env

# 安装依赖
pip install -r requirements.txt
```

3.  启动后端项目
```bash
python -m src.main
```

> ⚠️ **重要提示**: 如果是首次运行或升级到多会话版本，请先运行数据库修复脚本：
> ```bash
> python scripts/quick_fix.py
> ```
> 这将自动创建 `sessions` 和 `chat_messages` 表。

#### 前端启动

1. 进入 frontend 文件夹，打开一个新的命令行，输入以下命令：
```bash
npm install
```
2. 启动前端：
```bash
npm run dev
```
3. 在浏览器中访问 http://localhost:3000/ 即可使用本项目

