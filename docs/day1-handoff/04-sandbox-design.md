# 04. Sandbox Design

## `sandbox.py` 的职责

`sandbox.py` 的职责不是“执行任意 SQL”，而是：

- 只执行允许的只读 SQL
- 把执行过程放进单独子进程
- 给执行设置时间上限
- 统一返回结构化结果

换句话说，它是“训练和评估共用的 SQL 执行安全层”。

## 为什么不能直接在主进程里跑 SQL

如果直接在主进程里执行 SQL，会有几个问题：

- 某条 SQL 卡住时，不容易强制回收
- 错误和超时不容易统一处理
- agent 以后生成的 SQL 不可信，必须假设它会乱来

所以当前实现选择：

1. 主进程接收请求
2. 主进程把 SQL 做 base64 编码
3. 主进程启动一个新的 Python 子进程
4. 子进程真正连 SQLite 并执行 SQL
5. 子进程把结果以 JSON 打回主进程

## 现在有哪些防线

当前 `sandbox.py` 实际上有三层防线。

### 第一层：只读路径

SQLite 连接使用了只读 URI：

```text
file:xxx.sqlite?mode=ro
```

这层的作用是：

- 即使后面 SQL 写得很坏，也不能正常写入数据库

### 第二层：关键词和前缀限制

脚本会先判断 SQL 是否看起来像只读语句。

允许的前缀是：

- `select`
- `with`
- `explain select`
- `explain query plan`

拒绝的危险 token 包括：

- `drop`
- `insert`
- `update`
- `delete`
- `create`
- `alter`
- `attach`

注意一个 Day 1 里实际踩到的坑：

最开始如果用“字符串包含”判断，比如 `create` in SQL，就会误伤列名 `created`。

所以后来改成了“先分词，再判断 token 是否命中黑名单”。

### 第三层：SQLite authorizer

即使前面的字符串检查漏掉了什么，SQLite 自身还会再拦一层危险动作。

这层会拒绝：

- 建表
- 删表
- 插入
- 更新
- 事务
- pragma
- attach/detach

这层非常重要，因为它比字符串检查更接近数据库引擎本身。

## 超时怎么实现

超时不是靠 SQLite 自己，而是靠主进程调用子进程时的：

```python
subprocess.run(..., timeout=timeout_s)
```

当前默认值是：

- `3.0` 秒

超时后返回的不是崩溃，而是结构化结果：

- `ok = False`
- `error` 里写 `TimeoutExpired`

这样上层训练代码更容易直接消费。

## 返回值长什么样

`execute_sql(...)` 返回的是一个字典，大体上有这些字段：

- `ok`
- `rows`
- `columns`
- `row_count`
- `error`
- `duration_sec`

这比直接返回 Python 异常或原始 cursor 更适合：

- 做评估
- 记日志
- 给 reward 函数消费

## Day 1 修过的两个实际问题

### 问题 1：`created` 被误判成 `create`

原因：

- 最初黑名单判断过于粗糙

修复：

- 改成 token 级判断，而不是子串判断

### 问题 2：Spider 某些库里存在坏编码文本

原因：

- 个别 SQLite 文本列默认按 UTF-8 解码时会报错

修复：

- `sandbox.py` 让 SQLite 先按 `bytes` 读取
- `eval.py` 的归一层再显式做容错解码

这个设计的好处是：

- sandbox 层尽量“不丢数据”
- 评估层再做统一标准化

## 你接手时应该怎样理解 `sandbox.py`

不要把它理解成“一个命令行 demo”。

更准确的理解是：

“未来训练、验证、采样都会共用的数据库执行后端。”

如果未来要扩展：

- 多轮 agent rollout
- reward 计算
- traceback 回灌

大概率都还会复用这个文件。
