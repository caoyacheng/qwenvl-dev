# QwQ - 推理模型介绍

QwQ 是 Qwen 系列中的推理模型。相较于传统的指令调优模型，具备思考和推理能力的 QwQ 在下游任务中，尤其是解决难题时，能显著提升性能。

## QwQ-32B 模型概述

QwQ-32B 是一个中等规模的推理模型，其性能可与最先进的推理模型（如 DeepSeek-R1、o1-mini）相媲美。

## 本仓库包含的 QwQ 32B 模型特点

### 模型类型

- **因果语言模型**

### 训练阶段

- 预训练及后训练（监督微调和强化学习）

### 模型架构

- 带有 RoPE、SwiGLU、RMSNorm 和 Attention QKV 偏置的 transformers

### 参数信息

- **参数数量**：325 亿
- **非嵌入参数数量**：310 亿

### 模型结构

- **层数**：64 层
- **注意力头数（GQA）**：
  - Q 为 40 个
  - KV 为 8 个

### 上下文长度

- 完整支持 131,072 个 tokens
