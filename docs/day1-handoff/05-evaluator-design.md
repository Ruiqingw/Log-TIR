# 05. Evaluator Design

## `eval.py` 要解决什么问题

它要回答的问题不是：

- “模型生成的 SQL 和 gold SQL 长得像不像？”

而是：

- “模型生成的 SQL 执行后，结果和标准答案的结果是否一致？”

这就是 `execution match`。

## 为什么不用字符串直接比较 SQL

因为很多不同的 SQL 可以得到同一个正确答案。

例如下面两条 SQL，写法不同，但可能结果一样：

```sql
select count(*) from singer
```

```sql
select count(1) from singer
```

如果你只比较 SQL 字符串，就会把很多正确答案误判为错误。

所以 Day 1 选择比较“执行结果”，而不是比较“SQL 外观”。

## `eval.py` 的核心流程

可以拆成最细：

1. 找到 Spider 数据根目录
2. 找到 `dev.json`
3. 读取所有样本
4. 为每个样本找到对应数据库文件
5. 跑预测 SQL
6. 跑 gold SQL
7. 对两边执行结果做归一化
8. 判断是否匹配
9. 汇总 `total`、`matched`、`accuracy`
10. 把失败样本写到文件

## 结果比较时做了哪些归一化

### 1. `None` 统一处理

数据库里的空值统一表示为：

- `("null", None)`

### 2. bytes 先解码

因为某些 Spider 数据库会把坏编码文本读成 bytes，所以这里会做：

- `bytes -> utf-8 decode(errors="replace")`

### 3. 数字统一成 number

为了避免：

- `1`
- `1.0`

这种语义相同但类型不同的情况被误判，当前实现会把整数和浮点都归到：

- `("number", <rounded-float>)`

这属于“为了 reward 和评估稳定性做的工程性归一化”。

### 4. 字符串会去掉两端空白

例如：

- `"abc"`
- `" abc "`

归一后会更稳定。

## 为什么 `ORDER BY` 要特殊处理

不是所有 SQL 的结果都应该忽略顺序。

例如：

```sql
select name from players order by birth_date
```

这里顺序本身就是问题的一部分。

所以当前 evaluator 的策略是：

- 如果 SQL 包含 `ORDER BY`，按有序列表比较
- 如果没有 `ORDER BY`，按多重集合比较

这里用到的是 `Counter`，目的是：

- 保留重复行的数量
- 但忽略行顺序

## 为什么要先跑 gold SQL 自检

因为后面训练时如果某个样本拿不到 reward，你必须知道问题出在哪一层：

- 模型错了
- sandbox 错了
- evaluator 错了

如果连 gold SQL 自己都不能在 evaluator 上拿高分，那后面的训练结果就没有解释力。

所以 Day 1 把这件事当成硬门槛：

- Spider dev gold SQL 必须跑到 `>=95%`

实际结果比门槛更高：

- `1034 / 1034 = 1.0`

## 你接手时最应该记住的一点

当前 `eval.py` 不是“论文官方 evaluator 的完整复刻”，而是一个“为本地 agent 训练准备的工程可用 evaluator”。

它当前最重要的价值是：

- 够稳定
- 够透明
- 能直接接 reward
- 已被 gold SQL 校准过
