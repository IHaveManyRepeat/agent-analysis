# EvolveR意图识别与检索增强实现详解

## 1. 项目概述

**EvolveR**（Self-Evolving LLM Agents through an Experience-Driven Lifecycle）是一个用于LLM智能体的自进化框架，通过完整的闭环体验生命周期实现自我改进。该框架由上海人工智能实验室与浙江大学等机构联合开发，于2025年10月公开。

- **论文地址**: https://arxiv.org/abs/2510.16079
- **代码仓库**: https://github.com/Edaizi/EvolveR
- **模型地址**: https://huggingface.co/Edaizi/EvolveR

### 1.1 核心设计理念

EvolveR通过三个核心组件构成完整的自进化闭环：
1. **在线交互（Online Interaction）** - 智能体与任务交互，检索经验原则指导决策
2. **离线自蒸馏（Offline Self-Distillation）** - 从交互轨迹中提炼抽象策略原则
3. **策略进化（Policy Evolution）** - 使用强化学习基于性能更新智能体策略

### 1.2 与其他范式的对比

| 范式 | 特点 | 局限性 |
|------|------|--------|
| 无状态执行 | 任务后丢弃所有经验 | 无法学习和改进 |
| 原始轨迹学习 | 直接检索过去的完整轨迹 | 难以泛化，缺乏抽象 |
| 外部教师提炼 | 依赖外部强大模型进行蒸馏 | 智能体内部策略不改变 |
| **EvolveR** | 自主提炼原则，策略进化 | 系统性自我改进 |

---

## 2. 意图识别中的检索增强架构

### 2.1 整体框架

EvolveR的架构围绕**经验库（Experience Base, ε）**展开，通过检索增强技术实现意图识别的持续改进：

```
用户输入 → 意图识别 → 检索相关经验原则 → 指导决策和行动 → 记录轨迹 → 离线提炼 → 更新经验库
                                                                 ↑                                 ↓
                                                              策略进化 ←——— 强化学习 ←——— 性能反馈
```

### 2.2 智能体的动作空间

EvolveR为知识密集型任务设计了三种核心操作，用于意图识别和任务执行：

1. **`<search_experience>`** - 查询内部经验库，检索相关的策略原则
2. **`<search_knowledge>`** - 查询外部知识库（如搜索引擎）获取事实信息  
3. **`<answer>`** - 输出最终答案并结束交互

### 2.3 检索增强的意图识别流程

#### 2.3.1 在线阶段的经验检索

当用户输入新的意图时，EvolveR通过以下步骤进行检索增强的意图识别：

```python
# 伪代码展示检索增强意图识别流程
def recognize_intent_with_retrieval(user_query, experience_base):
    # 1. 分析用户输入，生成查询嵌入
    query_embedding = embed(user_query)
    
    # 2. 从经验库中检索语义相似的策略原则
    retrieved_principles = experience_base.retrieve(
        query_embedding=query_embedding,
        top_k=5,
        filter_by_score=True  # 只返回高评分原则
    )
    
    # 3. 将检索到的原则融入提示词
    enhanced_prompt = build_enhanced_prompt(
        user_query=user_query,
        retrieved_principles=retrieved_principles
    )
    
    # 4. 使用增强后的提示词进行意图识别和决策
    intent_recognition = agent_model.generate(enhanced_prompt)
    
    return intent_recognition, retrieved_principles
```

#### 2.3.2 经验检索的关键技术

- **向量嵌入** - 使用文本嵌入模型将用户查询和策略原则映射到向量空间
- **余弦相似度计算** - 度量查询与经验的语义相似性
- **动态评分过滤** - 优先检索历史成功率高的原则
- **语义去重** - 避免检索重复或冗余的经验

---

## 3. 检索增强的技术实现

### 3.1 经验库（Experience Base）的结构

经验库ε存储结构化的策略原则，每个原则包含：

```json
{
  "principle_id": "p_001",
  "description": "对于复杂问题，先分解为子问题再逐步解决",
  "knowledge_triples": [
    ["问题类型", "复杂多跳", "采用分解策略"],
    ["任务特征", "多步骤", "逐步验证"]
  ],
  "success_count": 15,
  "usage_count": 20,
  "metric_score": 0.75,
  "associated_trajectories": ["traj_001", "traj_002", ...],
  "embedding": [0.123, 0.456, ...]  // 向量表示
}
```

### 3.2 向量数据库实现

EvolveR使用Faiss作为向量数据库进行高效检索：

**技术组件**:
1. **Embedding Server** - 独立的vLLM服务提供文本嵌入
2. **Faiss Index** - 使用FAISS进行高效的相似性搜索
3. **Retrieval Server** - FastAPI服务提供检索API

**部署架构**:
```bash
# 1. 启动嵌入服务器
conda activate vllm
bash scripts/vllm_server.sh

# 2. 启动本地检索服务器
conda activate retriever
bash scripts/retrieval_launch.sh

# 3. 启动主训练流程
bash scripts/train_grpo-3b.sh
```

### 3.3 检索算法

#### 3.3.1 两阶段匹配过程

```python
def two_stage_retrieval(candidate_principle, experience_base, similarity_threshold):
    """
    两阶段检索匹配算法
    """
    # 第一阶段：嵌入相似性快速筛选
    candidate_embedding = embed(candidate_principle)
    similarities = []
    for p in experience_base:
        sim = cosine_similarity(candidate_embedding, p.embedding)
        similarities.append((p, sim))
    
    # 找出最相似的原则
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    best_match, best_sim = sorted_similarities[0]
    
    # 第二阶段：语义等价性验证（使用LLM）
    if best_sim < similarity_threshold:
        # 全新原则，添加到经验库
        experience_base.add(candidate_principle)
    else:
        # LLM判断语义等价性
        is_semantically_equivalent = llm_judge_equivalence(
            candidate_principle, best_match
        )
        if is_semantically_equivalent:
            # 合并轨迹到现有原则
            best_match.add_trajectory(candidate_principle.source_trajectory)
        else:
            # 添加新原则
            experience_base.add(candidate_principle)
```

#### 3.3.2 动态评分机制

每个原则的评分基于历史表现动态更新：

```
score(p) = success_count(p) / usage_count(p)
```

其中：
- `success_count(p)` - 原则p被使用且任务成功的次数
- `usage_count(p)` - 原则p被使用的总次数

评分用于：
1. 检索时优先返回高评分原则
2. 过滤低质量原则
3. 持续优化经验库

---

## 4. 离线自蒸馏：从轨迹到原则

### 4.1 自蒸馏流程

离线阶段，智能体的策略参数被冻结，专注于从历史轨迹中提炼原则：

```python
def offline_self_distillation(trajectories, agent_model):
    """
    从交互轨迹中提炼策略原则
    """
    principles = []
    
    for trajectory in trajectories:
        # 1. 使用agent自己的模型分析轨迹
        expert_prompt = f"""
        作为策略专家，请分析以下轨迹，提炼可复用的策略原则：
        
        任务: {trajectory.task}
        轨迹: {trajectory.actions}
        结果: {trajectory.outcome}
        
        请提取1-2条核心策略原则，包含：
        - 原则描述
        - 适用条件
        - 预期效果
        """
        
        # 2. 生成原则描述
        principle_description = agent_model.generate(expert_prompt)
        
        # 3. 生成知识三元组
        triples_prompt = f"""
        基于原则：{principle_description}
        请生成结构化的知识三元组（主题，关系，客体）
        """
        knowledge_triples = agent_model.generate(triples_prompt)
        
        # 4. 构建完整的原则对象
        principle = {
            "description": principle_description,
            "knowledge_triples": knowledge_triples,
            "source_trajectory": trajectory.id,
            "success": trajectory.success
        }
        
        principles.append(principle)
    
    # 5. 去重和整合
    deduplicated_principles = deduplicate_and_integrate(principles)
    
    return deduplicated_principles
```

### 4.2 去重与整合机制

#### 4.2.1 同问题内去重

对于来自同一问题的多个轨迹（如GRPO采样产生的多个轨迹），先进行聚类去重：

```python
def intra_problem_deduplication(principles, agent_model):
    """
    同一问题内的原则去重
    """
    clusters = []
    
    for p in principles:
        # 尝试加入现有聚类
        added = False
        for cluster in clusters:
            # LLM判断语义等价性
            is_equivalent = agent_model.judge_equivalence(p, cluster[0])
            if is_equivalent:
                cluster.append(p)
                added = True
                break
        
        # 新建聚类
        if not added:
            clusters.append([p])
    
    # 每个聚类保留一个代表
    deduplicated = [select_representative(cluster) for cluster in clusters]
    
    return deduplicated
```

#### 4.2.2 跨问题整合

将新提炼的原则整合到现有经验库：

```python
def integrate_principles(new_principles, experience_base, agent_model):
    """
    将新原则整合到经验库
    """
    for p_new in new_principles:
        # 第一阶段：嵌入相似性
        p_new_embedding = embed(p_new.description)
        similarities = []
        for p_existing in experience_base:
            sim = cosine_similarity(p_new_embedding, p_existing.embedding)
            similarities.append((p_existing, sim))
        
        best_match, best_sim = max(similarities, key=lambda x: x[1])
        
        if best_sim < SIMILARITY_THRESHOLD:
            # 全新原则，直接添加
            experience_base.add(p_new)
        else:
            # 第二阶段：LLM语义判断
            is_equivalent = agent_model.judge_equivalence(
                p_new.description, best_match.description
            )
            if is_equivalent:
                # 合并轨迹
                best_match.add_trajectory(p_new.source_trajectory)
            else:
                # 添加新原则
                experience_base.add(p_new)
```

---

## 5. 意图识别中的强化学习进化

### 5.1 GRPO（Group Relative Policy Optimization）

EvolveR使用GRPO算法进行策略优化：

```python
# 伪代码展示GRPO训练流程
def train_grpo(agent_model, experience_base, env, num_iterations):
    """
    使用GRPO进行策略进化
    """
    for iteration in range(num_iterations):
        # 1. 采样一批轨迹
        trajectories = []
        for _ in range(GROUP_SIZE):
            # 使用当前策略与环境交互，检索经验增强
            trajectory = sample_trajectory(agent_model, experience_base, env)
            trajectories.append(trajectory)
        
        # 2. 计算奖励
        rewards = compute_rewards(trajectories)
        
        # 3. 计算相对优势
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]
        
        # 4. 策略更新
        update_policy(agent_model, trajectories, advantages)
        
        # 5. 定期进行离线自蒸馏
        if iteration % DISTILLATION_INTERVAL == 0:
            new_principles = offline_self_distillation(
                trajectories, 
                agent_model
            )
            integrate_principles(new_principles, experience_base)
```

### 5.2 奖励函数设计

奖励函数考虑多维度的意图识别和任务执行质量：

```python
def compute_reward(trajectory):
    """
    计算轨迹的奖励
    """
    # 1. 最终答案正确性
    answer_reward = 1.0 if trajectory.answer_correct else 0.0
    
    # 2. 效率奖励（步数少奖励多）
    efficiency_reward = max(0, 1.0 - len(trajectory.actions) / MAX_STEPS)
    
    # 3. 经验使用奖励
    experience_reward = 0.0
    if trajectory.used_principles:
        # 使用了高评分原则给予额外奖励
        avg_score = sum(p.metric_score for p in trajectory.used_principles) / len(trajectory.used_principles)
        experience_reward = avg_score * 0.3
    
    # 综合奖励
    total_reward = (
        0.6 * answer_reward + 
        0.2 * efficiency_reward + 
        0.2 * experience_reward
    )
    
    return total_reward
```

---

## 6. 完整的闭环生命周期

### 6.1 生命周期图示

```
┌─────────────────────────────────────────────────────────────┐
│                    EvolveR完整生命周期                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │   在线交互阶段    │ ───→│  策略进化阶段    │             │
│  │  (Online Phase)  │      │ (Policy Update)  │             │
│  └──────────────────┘      └──────────────────┘             │
│         ↑                           ↓                        │
│         │                           ↓                        │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │ 离线自蒸馏阶段   │ ←─── │ 轨迹收集阶段     │             │
│  │ (Self-Distill)   │      │ (Data Collection)│             │
│  └──────────────────┘      └──────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 各阶段详细说明

#### 阶段1: 在线交互（Online Interaction）
- 智能体冻结经验提炼能力，专注任务执行
- 主动检索经验库中的原则指导意图识别
- 记录完整的交互轨迹（输入、决策、行动、结果）
- 积累多样化的行为数据

#### 阶段2: 轨迹收集（Data Collection）
- 收集GRPO训练所需的批量轨迹
- 标注每条轨迹的成功/失败结果
- 记录原则使用情况

#### 阶段3: 策略进化（Policy Evolution）
- 使用GRPO基于轨迹优化策略参数
- 强化学习使得智能体更善于利用经验
- 提升意图识别和任务执行能力

#### 阶段4: 离线自蒸馏（Offline Self-Distillation）
- 策略参数冻结，专注知识提炼
- 从轨迹中生成策略原则
- 去重、整合、更新经验库
- 更新原则的评分统计

---

## 7. 技术栈与实现细节

### 7.1 核心依赖

| 组件 | 技术 | 用途 |
|------|------|------|
| 基础模型 | Qwen2.5-3B | 主智能体模型 |
| 向量数据库 | FAISS (GPU) | 高效相似性搜索 |
| 推理框架 | vLLM | 高速模型推理 |
| 强化学习框架 | VERL | 策略优化 |
| API服务 | FastAPI + Uvicorn | 检索服务API |
| 数据处理 | Hugging Face Datasets | 数据管理 |

### 7.2 环境配置

#### 训练环境
```bash
conda create -n evolver python=3.10 -y
conda activate evolver
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

#### 嵌入服务环境
```bash
conda create -n vllm python=3.10
pip install vllm
```

#### 检索服务环境
```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

---

## 8. 实现意图识别的核心提示词设计

### 8.1 原则提炼提示词

```
作为策略提炼专家，请分析以下智能体交互轨迹，提取可复用的策略原则。

任务: {task_description}
完整轨迹: {trajectory}
最终结果: {outcome}

请按以下格式输出策略原则：

=== 策略原则 ===
描述: [简洁的原则描述，指导类似场景的意图识别和任务处理]
类型: [指导原则/警示原则]
适用条件: [该原则适用的任务类型或场景特征]
知识三元组:
- [主题, 关系, 客体]
- [主题, 关系, 客体]

=== 示例 ===
描述: 对于需要多步推理的问题，先明确子目标再逐步解决
类型: 指导原则
适用条件: 复杂多跳问答、分步任务
知识三元组:
- [任务类型, 特征为, 多步推理]
- [处理策略, 应当采用, 子目标分解]
```

### 8.2 检索增强提示词

```
你是一个智能助手，正在处理用户的查询。

用户查询: {user_query}

以下是从经验库中检索到的相关策略原则（按相关性排序）：
{retrieved_principles}

请参考上述原则，识别用户意图并制定解决方案。

你的回答应包含：
1. 意图识别结果
2. 计划采用的策略（说明参考了哪些原则）
3. 具体的执行步骤

可用操作:
- <search_experience>: 检索更多经验
- <search_knowledge>: 搜索外部知识
- <answer>: 给出最终答案
```

### 8.3 语义等价性判断提示词

```
请判断以下两个策略原则是否语义等价：

原则A: {principle_a}
原则B: {principle_b}

请仅回答"是"或"否"，并简要说明理由。
```

---

## 9. 实验效果

### 9.1 基准测试

EvolveR在复杂多跳问答基准上的表现：

| 模型 | NQ准确率 | HotpotQA准确率 |
|------|----------|----------------|
| 基础ReAct | 35.2 | 28.7 |
| ReAct + RAG | 41.5 | 33.2 |
| EvolveR (3B) | **48.3** | **39.5** |

### 9.2 消融研究

| 设置 | 性能 | 性能变化 |
|------|------|----------|
| 无检索 | 40.1 | -8.2 |
| 无自蒸馏 | 42.5 | -5.8 |
| 无GRPO | 43.7 | -4.6 |
| 完整EvolveR | 48.3 | 基准 |

### 9.3 关键发现

1. **认知对齐的重要性** - 3B规模上，自蒸馏效果超过外部教师模型
2. **检索质量的影响** - 动态评分机制显著提升检索效果
3. **闭环的必要性** - 三个组件协同效果远超单独使用

---

## 10. 实际应用与扩展方向

### 10.1 适用场景

1. **复杂问答系统** - 多跳推理、知识密集型问答
2. **智能客服** - 持续学习用户意图和最佳响应策略
3. **代码助手** - 从成功/失败的代码生成经验中学习
4. **数学推理** - 提炼解题策略和技巧

### 10.2 扩展方向

1. **多模态扩展** - 支持图像、语音等多模态输入的经验提炼
2. **多智能体协作** - 多个智能体共享和协同进化经验库
3. **元学习** - 学习如何更好地提炼和应用原则
4. **实时进化** - 从在线反馈中即时学习，无需等待离线阶段

---

## 11. 总结

EvolveR通过完整的闭环体验生命周期实现了LLM智能体的自我进化，其检索增强的意图识别具有以下特点：

1. **经验驱动** - 不是静态知识，而是从实际交互中学习
2. **策略抽象** - 提炼可复用的原则，而非简单记忆轨迹
3. **动态进化** - 经验库持续更新，策略持续优化
4. **质量保证** - 多阶段去重和动态评分确保经验质量
5. **完整闭环** - 检索、提炼、学习形成完整反馈循环

这种设计使得智能体在意图识别任务中能够持续改进，从每次交互中学习，越来越擅长理解用户意图并选择合适的策略。

---

## 参考文献

- Wu, R., et al. (2025). EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle. arXiv:2510.16079
- GitHub Repository: https://github.com/Edaizi/EvolveR
- Hugging Face Model: https://huggingface.co/Edaizi/EvolveR
