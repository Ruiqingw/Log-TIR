# Self-Correction Rollout

## 相关文件

- `agent_rollout.py`
- `openrlhf_agent.py`
- `openrlhf_reward.py`
- `sandbox.py`
- `tests/test_agent_rollout.py`
- `tests/test_openrlhf_agent.py`

## 项目想要的自纠错是什么

项目核心 thesis 之一是：

1. 模型读 schema 和问题。
2. 模型生成 SQL。
3. sandbox 执行 SQL。
4. 如果 SQL 报错、空结果、或结果不匹配，把反馈写回 prompt。
5. 模型最多再试一次。

这个过程可以称为 multi-turn self-correction。

## 独立 Helper

`agent_rollout.py` 提供本地可测试的 helper：

```python
rollout_self_correction(
    prompt,
    label,
    model_fn,
    max_turns=2,
    timeout_s=3.0,
)
```

它适合做离线分析和单元测试：传入一个 `model_fn(prompt) -> response`，helper 会执行 SQL、生成 feedback，并在第二轮重新调用 `model_fn`。

## OpenRLHF Multi-Turn Agent

`openrlhf_agent.py` 是训练时使用的 OpenRLHF agent function。

它实现了 OpenRLHF 文档要求的接口：

- `AgentInstance.reset(states)`：接收初始 observation/prompt。
- `AgentInstance.step(states)`：接收本轮 `action_text` 和 `label`，执行 SQL，返回 reward、feedback 和 done。
- `AgentExecutor(MultiTurnAgentExecutor)`：把 `AgentInstance` 注册给 OpenRLHF。

GRPO 脚本默认启用多轮：

```bash
GRPO_MULTI_TURN=1 START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

脚本会传：

```text
--agent_func_path openrlhf_agent.py
```

如果需要回退到原来的单轮 reward function：

```bash
GRPO_MULTI_TURN=0 START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

回退模式会传：

```text
--reward.remote_url openrlhf_reward.py
```

## 训练时每轮发生什么

第一轮：

1. OpenRLHF 把 prompt 作为 observation。
2. 模型生成 `<thought>` 和 `<action>`。
3. `openrlhf_agent.py` 解析 `<action>`。
4. `sandbox.py` 执行 SQL。
5. `openrlhf_reward.py` 计算 format/no-error/exec-match。

如果第一轮已经 execution match：

- episode 结束。
- reward 是 `1.3`。
- 不再生成第二轮。

如果第一轮没匹配，但还没达到 `LOGTIR_AGENT_MAX_TURNS=2`：

- agent 返回 `environment_feedback`。
- OpenRLHF 把 feedback 拼回上下文。
- 模型生成第二轮 response。

第二轮：

- 如果修正成功，reward 是 `1.3`。
- 如果格式正确且 SQL 可执行但结果错，reward 是 `0.3`。
- 如果格式错，reward 是 `0.0`。
- 第二轮后无论是否成功都结束。

## 反馈是怎么构造的

如果 SQL 执行报错，反馈大致是：

```text
Execution feedback:
The previous SQL failed in SQLite.
Error: OperationalError: no such table: missing
Revise the SQL and answer again with exactly <thought> and <action>.
```

如果 SQL 执行成功但结果为空：

```text
Execution feedback:
The previous SQL executed but returned an empty result.
Check whether joins or filters are too restrictive, then answer again with exactly <thought> and <action>.
```

如果 SQL 执行成功但不匹配 gold 结果：

```text
Execution feedback:
The previous SQL executed but did not match the expected result.
Revise the selected tables, joins, filters, aggregation, ordering, or DISTINCT decision, then answer again with exactly <thought> and <action>.
```

## 为什么最多 2 轮

项目默认 self-correction budget 是最多 2 turns。

原因：

- 太多轮会显著增加 rollout 成本。
- 训练时每个 prompt 已经有 `G=4` 多样本采样。
- 两轮足够展示“读取执行反馈并修正”的 agentic 行为。
- 对简历项目来说，2 轮比无限循环更容易解释和复现。

## 已覆盖的测试

`tests/test_agent_rollout.py` 覆盖：

- 独立 helper 第一轮 SQL 报错时进入第二轮。
- 第二轮 prompt 中包含 `Execution feedback`。
- SQLite 错误文本会进入第二轮 prompt。
- 第一轮正确时提前停止。
- 相对 `db_path` 能通过 `SPIDER_ROOT` 解析。

`tests/test_openrlhf_agent.py` 覆盖：

- OpenRLHF agent 第一轮 SQL 报错时返回 execution feedback。
- 第二轮修正后 episode 结束并返回 `1.3`。
- 第二轮仍错误时 episode 结束并返回 shaped reward，例如 `0.3`。

## 剩余风险

代码层面已经接入 `--agent_func_path`，但还需要在服务器上做 smoke test，因为 OpenRLHF 的 agent API、参数名和安装版本可能变化。

服务器验证重点：

- `openrlhf_agent.py` 是否能被 Ray worker import。
- `states` 里是否使用文档中的 `observation`、`action_text`、`label` 字段。
- `environment_feedback` 是否确实进入下一轮上下文。
- 多轮模式是否和当前 GRPO 参数组合兼容。
