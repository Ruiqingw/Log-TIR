# 07. Rerun Playbook

这一页是给“我不想先看所有背景，只想确认怎么从头重跑一遍”的人准备的。

## 最短重跑路径

在仓库根目录执行：

```bash
python3 -m py_compile sandbox.py eval.py tests/test_sandbox.py tests/test_eval.py
python3 -m pytest tests -q
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
```

如果这四条都过了，就说明 Day 1 当前状态基本完整。

## 从空白环境开始的完整路径

### 第 1 步：确认你在仓库根目录

```bash
pwd
ls -la
```

你应该能看到：

- `CLAUDE.md`
- `AGENTS.md`
- `sandbox.py`
- `eval.py`
- `tests/`

### 第 2 步：准备 Spider 数据

如果 `data/spider/spider_data/dev.json` 已存在，可以跳过。

否则执行：

```bash
mkdir -p data
python3 -m pip install -U gdown
python3 -m gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/spider_data.zip
unzip -q data/spider_data.zip -d data/spider
```

### 第 3 步：确认关键数据文件存在

```bash
find data/spider -maxdepth 2 -type f | sort
find data/spider -type f -name '*.sqlite' | sed -n '1,10p'
```

你至少应该确认：

- `data/spider/spider_data/dev.json`
- `data/spider/spider_data/database/...`

### 第 4 步：做静态语法检查

```bash
python3 -m py_compile sandbox.py eval.py tests/test_sandbox.py tests/test_eval.py
```

这一步的作用是先挡掉最基本的语法错误。

### 第 5 步：跑单元测试

```bash
python3 -m pytest tests -q
```

Day 1 版本期望看到：

```text
.........                                                                [100%]
9 passed
```

时间数字可能略有变化，不重要。

### 第 6 步：跑 100 条 smoke test

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

期望结果：

- `total = 100`
- `matched = 100`
- `accuracy = 1.0`

### 第 7 步：跑 Spider dev 全量自检

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
```

Day 1 当前版本期望结果：

- `total = 1034`
- `matched = 1034`
- `accuracy = 1.0`

同时：

- `spider_gold_failures.json` 应为空列表

## 如果某一步失败，优先检查什么

### 情况 1：找不到 Spider 数据

先检查：

- `data/spider/spider_data/dev.json`
- `data/spider/spider_data/database/`

### 情况 2：pytest 失败

先看是哪一类：

- sandbox 失败
- evaluator 失败

再看最近是否改过：

- SQL 黑名单判断
- `ORDER BY` 逻辑
- bytes 解码
- 数值归一

### 情况 3：100 条能过，全量不过

优先怀疑：

- 某个特殊数据库的编码问题
- 路径解析遗漏
- 某个边界数据类型未归一

这时先打开：

- `spider_gold_failures.json`

## 一句话交接版本

如果你要把 Day 1 状态转述给另一个人，可以直接说：

“Spider 数据、SQL sandbox、execution-match evaluator、单测和 gold 自检都已经完成；现在可以安全地进入 SFT 数据和训练脚本阶段。”
