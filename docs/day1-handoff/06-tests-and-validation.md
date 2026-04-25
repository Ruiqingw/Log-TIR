# 06. Tests And Validation

## 为什么 Day 1 不能只靠“看起来能跑”

Day 1 做的是基础设施。

基础设施最怕的不是“明显坏掉”，而是“看起来没问题，但边界条件错了”。

例如：

- `DROP TABLE` 没被拦住
- 某条 SQL 超时后直接把流程卡死
- `ORDER BY` 被忽略，导致评估假阳性
- `1` 和 `1.0` 被错判为不同

这些问题如果不在 Day 1 发现，后面训练时会非常难排查。

## 现在有哪些测试文件

### `tests/test_sandbox.py`

它覆盖了这些行为：

1. 正常只读查询可以执行
2. `DROP TABLE` 会被拒绝
3. 语法错误会返回结构化错误
4. 列名 `created` 不会被误伤成危险 `create`
5. 超时会返回结构化错误，而不是把程序拖住

### `tests/test_eval.py`

它覆盖了这些行为：

1. bytes 会被正确解码
2. `ORDER BY` 会影响是否匹配
3. `1` 和 `1.0` 会被视为等价
4. 一个最小 Spider 风格数据集可以被 evaluator 正确跑通

## 为什么测试不用真实 Spider 全量数据

单测的目标不是“重复做一次大评测”，而是：

- 快速
- 小规模
- 可控
- 容易定位失败原因

所以测试里都在临时目录创建小 SQLite 数据库和小 JSON 样本。

这样做的好处是：

- 不依赖大数据下载是否完成
- 跑起来很快
- 失败时更容易知道是哪个逻辑坏了

## 现在有哪些真实验证结果

除了 pytest，还做了两类真实运行验证。

### 验证 1：Spider dev 前 100 条 gold SQL

命令：

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

结果：

```json
{
  "total": 100,
  "matched": 100,
  "accuracy": 1.0
}
```

### 验证 2：Spider dev 全量 1034 条 gold SQL

命令：

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
```

结果：

```json
{
  "total": 1034,
  "matched": 1034,
  "accuracy": 1.0
}
```

同时：

- `spider_gold_failures.json` 是空列表

这说明至少在当前 Day 1 版本里：

- 数据路径是对的
- sandbox 能支撑 Spider 执行
- evaluator 的比较逻辑没有明显系统性错误

## 实际运行记录在哪里

如果你需要看最原始的“跑过什么命令、得到什么结果”，请看：

- `.codex-reports/day1.md`

这个文件比本说明更像“实验记录”。

## 你接手时应该怎么使用这些验证

如果你只是改了文档，不需要重跑。

如果你改了下面任意一类内容，应该重跑：

- `sandbox.py`
- `eval.py`
- SQL 结果归一逻辑
- Spider 数据路径解析逻辑

最少重跑：

```bash
python3 -m pytest tests -q
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

如果你要提交一个更关键的改动，建议重跑全量：

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
```
