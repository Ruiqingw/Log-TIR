# Reward Function

## 相关文件

- `openrlhf_reward.py`
- `sandbox.py`
- `eval.py`
- `sft_data.py`
- `tests/test_openrlhf_reward.py`

## Reward 的目标

Text-to-SQL 的最终目标是执行结果正确，也就是 execution match。

但是如果只给完全正确的 SQL 打分，训练早期 reward 会太稀疏。Day 3 使用三层 reward：

```text
format = 0.1
no-error = 0.2
exec-match = 1.0
```

最高分是：

```text
0.1 + 0.2 + 1.0 = 1.3
```

## 三层 reward 分别是什么意思

`format = 0.1`：

- 模型输出能被解析成 `<thought>` 和 `<action>`。
- `<action>` 里可以抽出 SQL。
- 这一步不执行 SQL，只检查格式。

`no-error = 0.2`：

- `<action>` 中的 SQL 能在 SQLite sandbox 里执行成功。
- 没有语法错误。
- 没有访问不存在的表或列。
- 没有被只读 sandbox 拒绝。
- 没有超时。

`exec-match = 1.0`：

- 模型 SQL 的执行结果和 gold SQL 的执行结果一致。
- 对 `ORDER BY` 查询保留行顺序。
- 对非 `ORDER BY` 查询按多重集合比较，避免无序结果导致误判。
- 复用 Day 1 的 normalization 逻辑，降低数字和字符串格式噪声。

## 打分流程

`score_response(response, label)` 的流程：

1. 解析 `label`。
2. 解析 response 的 `<thought>` 和 `<action>`。
3. 格式解析成功，加 `0.1`。
4. 根据 label 找到 SQLite 数据库路径。
5. 用 `sandbox.execute_sql` 执行模型 SQL。
6. 执行成功，加 `0.2`。
7. 执行 gold SQL。
8. 对比模型结果和 gold 结果。
9. 如果 execution match，加 `1.0`。
10. 返回 reward 和诊断字段。

返回结果示例：

```json
{
  "reward": 1.3,
  "format_match": true,
  "no_error": true,
  "exec_match": true,
  "error": ""
}
```

## OpenRLHF 调用入口

OpenRLHF 需要的是 `reward_func(queries, prompts, labels)`。

Day 3 实现的函数会：

1. 遍历 OpenRLHF 传入的 `queries`、`prompts`、`labels`。
2. 如果 query 以 prompt 开头，就把 prompt 从 query 中去掉，只保留模型 response。
3. 调用 `score_response`。
4. 收集 reward。
5. 返回 `torch.tensor(scores, dtype=torch.float32)`。

这个形状符合 OpenRLHF 的 reward function 约定。

## 为什么要缓存 gold SQL

GRPO 会对同一个 prompt 采样多条 response。比如 `G=4` 时，同一个问题会有 4 条 SQL 候选。

如果每条候选都重新执行一次 gold SQL，会浪费时间：

```text
同一个 prompt 的 4 个 rollout -> gold SQL 重复执行 4 次
```

Day 3 用：

```python
@lru_cache(maxsize=8192)
def _execute_gold_sql_cached(db_path, gold_sql, timeout_s):
    ...
```

这样同一个 `(db_path, gold_sql, timeout_s)` 只执行一次。后续候选复用缓存结果。

## 已覆盖的测试

`tests/test_openrlhf_reward.py` 覆盖：

- 完全 execution match 得 `1.3`。
- 格式正确、SQL 无错误、但结果错误时得 `0.3`。
- 格式错误时得 `0.0`。
- label 可以是 JSON 字符串。
- 相对 `db_path` 能通过 `SPIDER_ROOT` 解析。
- gold SQL 执行缓存确实减少重复调用。

## 需要注意的风险

`no-error` 不能给太高。

如果 `no-error` reward 太高，模型可能学会输出能执行但无意义的 SQL，例如：

```sql
SELECT 1
```

所以当前权重让 `exec-match = 1.0` 占主导，`format` 和 `no-error` 只作为早期训练的稠密信号。

