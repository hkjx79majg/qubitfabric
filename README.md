# QubitFabric

这是一个面向量子-经典混合计算的量子-经典混合计算的编排与仿真平台。长期目标是提供量子电路 IR 与等价变换、参数化变分电路、噪声模型与含噪仿真、梯度估计、经典优化器调度、任务切分与卸载决策、混合运行时状态机与断点续算，把量子-经典协同执行沉淀为可复用服务。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m qubitfabric.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `QUBITFABRIC_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 电路 IR

`qubitfabric.service.Service` 提供可序列化量子电路 IR 的公开方法，输入输出均为 JSON 原生类型，不依赖第三方量子 SDK：

- `normalize(spec)`：校验并规范化电路对象（`qubit_count` 为非负整数，可选 `parameters` 参数声明与 `operations` 操作序列；支持 `x`、`h`、`rx`、`rz`、`cx` 门，旋转角可为有限实数或已声明参数的线性表达式）。成功时返回规范化的新对象，不修改输入。
- `simplify(circuit)`：确定性等价变换——删除相邻同位自逆门对（`x`/`h`/`cx`）、合并无同位干扰的同轴 `rx`/`rz` 旋转、常量角规约到 `[-pi, pi)` 且归零删除。返回 `{"circuit", "removed_operations", "merged_operations"}`，重复简化结果不变。
- `bind(circuit, bindings)`：把参数映射替换为有限实数角度，全部绑定后不再残留参数声明。

校验失败抛出公开的 `CircuitValidationError`（含稳定 `code` 与指向错误位置的 `path`，只报告按输入顺序遇到的首个错误）；绑定失败抛出 `ParameterBindingError`（`code` 为 `missing_parameter` 或 `unknown_parameter`，同类多个名称按字典序取首个）。两个异常也可从 `qubitfabric` 包顶层导入。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线刻意不包含量子电路 IR、变分电路、噪声模型与混合执行的实现，以便后续任务从已冻结事实出发独立设计并验证这些能力。
