# nanobot 架构分析

## 1. 职责概述

nanobot 是一个开源的超轻量级 AI 代理，继承了 OpenClaw、Claude Code 和 Codex 的精神。它的主要职责包括：

- **轻量级代理核心**：保持核心代理循环小而可读，同时支持聊天通道、内存、MCP 和实际部署路径
- **多平台支持**：支持 Telegram、Discord、WeChat、Feishu、Slack、Email、QQ 等多种聊天平台
- **多提供商集成**：支持 OpenRouter、OpenAI、Anthropic、Moonshot/Kimi、GLM、MiniMax 等多种 LLM 提供商
- **MCP 支持**：集成 Model Context Protocol，扩展代理能力
- **内存系统**：支持持久化内存和会话管理
- **WebUI 界面**：提供网页界面进行交互
- **安全设计**：内置安全沙箱和权限控制
