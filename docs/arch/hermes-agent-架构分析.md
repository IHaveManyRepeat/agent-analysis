# hermes-agent 架构分析

## 1. 职责概述

hermes-agent 是一个自我改进的 AI 代理，由 Nous Research 开发。它的主要职责包括：

- **自改进学习循环**：从经验中创建技能，在使用过程中持续改进，通过定期提示来维护记忆
- **多平台通信**：支持 Telegram、Discord、Slack、WhatsApp、Signal 和 CLI 等多种通信渠道
- **智能记忆管理**：FTS5 会话搜索，LLM 摘要，跨会话回忆，Honcho 对话用户建模
- **调度自动化**：内置 cron 调度器，支持跨平台交付
- **并行工作流**：生成隔离的子代理进行并行工作
- **灵活部署**：支持本地、Docker、SSH、Daytona、Singularity 和 Modal 等多种部署方式
- **研究就绪**：批量轨迹生成，Atropos RL 环境，轨迹压缩
