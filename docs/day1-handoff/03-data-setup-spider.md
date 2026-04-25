# 03. Data Setup: Spider

## 为什么选 Spider

项目最初的坏路线是用“合成个人日志”做数据。

这个路线的问题在于：

- 数据太小
- 任务分布太窄
- 外部效度很差
- 面试时很难说明这个 benchmark 为什么可信

Spider 更适合作为项目起点，因为它：

- 是公开、常见、可复现的学术基准
- 自带多个领域的数据库
- 每个样本都有自然语言问题和标准 SQL
- 数据库本身可以本地 SQLite 执行

## 现在数据放在哪里

Day 1 下载后的数据位置如下：

- 压缩包：`data/spider_data.zip`
- 解压根目录：`data/spider/spider_data/`
- dev 文件：`data/spider/spider_data/dev.json`
- 数据库目录：`data/spider/spider_data/database/`

这个目录结构很重要，因为 `eval.py` 依赖它去定位：

- `dev.json`
- 每个 `db_id` 对应的 `.sqlite` 文件

## Day 1 做数据这一步时，实际发生了什么

步骤可以拆成最细：

1. 创建 `data/` 目录
2. 安装 `gdown`
3. 从 Spider 官方 Google Drive 链接下载数据压缩包
4. 解压到 `data/spider/`
5. 检查是否存在：
   - `data/spider/spider_data/dev.json`
   - `data/spider/spider_data/database/...`
6. 用 `eval.py` 实际读取这些路径，确认代码不是“假设路径正确”

## 为什么路径解析要写得更稳

Spider 的压缩包解压后，经常不是你想象中的：

- `data/spider/dev.json`

而是：

- `data/spider/spider_data/dev.json`

如果评估脚本把路径写死，第一次下载官方数据就会出错。

因此 `eval.py` 做了多候选路径解析：

- 先试 `data/spider/`
- 再试 `data/spider/spider/`
- 再试 `data/spider/spider_data/`

这不是“追求优雅”，而是为了让脚本对真实数据包的目录层级更健壮。

## 如果你要重新下载 Spider

最短流程如下：

```bash
python3 -m pip install -U gdown
python3 -m gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/spider_data.zip
unzip -q data/spider_data.zip -d data/spider
```

然后立刻检查：

```bash
find data/spider -maxdepth 2 -type f | sort
find data/spider -type f -name '*.sqlite' | sed -n '1,10p'
```

## 你接手时最应该确认的两件事

1. `dev.json` 是否还在 `data/spider/spider_data/`
2. `database/<db_id>/<db_id>.sqlite` 是否完整

如果这两件事成立，Day 1 的数据层通常就没问题。
