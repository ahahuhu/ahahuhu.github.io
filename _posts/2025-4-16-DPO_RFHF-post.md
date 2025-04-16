---
title: 'RLHF VS DPO'
date: 2025-4-16
permalink: 
tags:
  - deep learning
  - nlp
  - post-training
---
## RLHF

RLHF (Reinforcement Learning from Human Feedback)

**RLHF 是什么？**

RLHF 是一种用于微调大型语言模型（LLM）的技术，旨在使模型的输出更符合人类的偏好和期望。传统的监督学习（Supervised Fine-tuning, SFT）主要让模型模仿特定格式或内容的文本，但很难捕捉到人类对于“好”的回答所具有的细微差别、主观感受（例如有用性、无害性、诚实性）。RLHF 通过引入人类的反馈信号（通常是比较形式的偏好数据），训练一个奖励模型（Reward Model, RM），然后利用这个奖励模型作为强化学习（RL）环境中的奖励信号，来进一步优化 LLM。

**RLHF 的目的**

1.  **对齐（Alignment）**: 主要目标是让 LLM 的行为与人类的价值观和偏好对齐。这意味着模型应该生成：
    *   **有用的（Helpful）**: 能准确、相关地回答问题或完成指令。
    *   **诚实的（Honest）**: 不捏造信息，在不确定时承认。
    *   **无害的（Harmless）**: 避免生成有偏见、歧视性、有毒或不安全的内容。
2.  **提升主观质量**: 改善 SFT 难以量化的输出质量，如写作风格、幽默感、创造力等。
3.  **增强安全性**: 减少模型产生有害输出的可能性。
4.  **提高指令遵循能力**: 让模型更可靠地遵循用户的复杂或细微指令。

**RLHF 的核心流程**

RLHF 通常包含三个主要阶段：

**阶段 1: 监督微调 (Supervised Fine-tuning, SFT)**

1.  **目标**: 让预训练的 LLM 具备基本的指令遵循能力和对话能力。
2.  **数据**: 高质量的 "指令-回答" 对 (Prompt-Response pairs)。这些数据通常由人工标注者编写，或者来自高质量的公开数据集。
3.  **过程**: 使用标准的监督学习方法，在预训练的 LLM 基础上进行微调，最小化模型预测的回答与标准答案之间的差异（通常是交叉熵损失）。
4.  **结果**: 得到一个 SFT 模型 (π_SFT)。这个模型是后续 RLHF 阶段的基础。

**阶段 2: 训练奖励模型 (Reward Model, RM)**

1.  **目标**: 训练一个模型，使其能够根据人类偏好对 LLM 生成的回答进行打分。
2.  **数据**: 人类偏好比较数据。
    *   给定一个指令 (prompt, x)。
    *   使用 SFT 模型（或其他模型）针对该指令生成多个不同的回答 (responses, y)。例如，生成回答 y1, y2, y3, y4。
    *   让人类标注者对这些回答进行排序，指出哪个更好。通常是成对比较 (pairwise comparison)，例如标注者认为 y_w (winner) 比 y_l (loser) 更好。
    *   收集大量的 (x, y_w, y_l) 数据点。
3.  **模型**: RM 通常基于 SFT 模型（或其副本），移除最后的输出层，替换为一个输出单个标量值（奖励分数）的线性层。
4.  **过程**: 训练 RM 预测哪个回答更受人类偏好。对于一对回答 (y_w, y_l)，RM 应该给 y_w 更高的分数 R(x, y_w)，给 y_l 更低的分数 R(x, y_l)。
5.  **损失函数**: 通常使用 **Bradley-Terry 模型** 思想的 Pairwise Ranking Loss。目标是最大化被偏好的回答 (y_w) 的分数与不被偏好的回答 (y_l) 的分数之差的 Sigmoid 值。对应的损失函数（最小化负对数似然）是：

    $$
    Loss(θ) = - E_{(x, y_w, y_l) \in D} [ log( σ( R_θ(x, y_w) - R_θ(x, y_l) ) ) ]
    $$

    *   $ D $: 人类偏好数据集。
    *   $ R_θ(x, y) $: 奖励模型（参数为 θ）对给定输入 x 和回答 y 输出的标量分数。
    *   $ σ $: Sigmoid 函数 (σ(z) = 1 / (1 + exp(-z)))。
    *   $ E $: 表示期望值，即对数据集 D 中的所有样本计算平均损失。
    *   这个损失函数鼓励模型参数 θ 学习到使 $ R_θ(x, y_w) $ 显著大于 $ R_θ(x, y_l) $。

6.  **结果**: 得到一个训练好的 RM，可以为任意 (指令, 回答) 对打分，分数越高表示越符合人类偏好。

**阶段 3: 基于强化学习的微调 (RL Fine-tuning)**

1.  **目标**: 使用 RM 作为奖励信号，通过 RL 算法进一步优化 SFT 模型，使其生成能获得更高奖励分数的回答。
2.  **框架**:
    *   **策略 (Policy)**: 当前需要优化的 LLM，通常从 SFT 模型 (π_SFT) 初始化，记为 π_RL。
    *   **动作空间 (Action Space)**: LLM 可能生成的所有词汇序列（即回答）。
    *   **状态 (State)**: 当前的指令 (prompt, x)。
    *   **奖励函数 (Reward Function)**: 由训练好的 RM 提供。对于策略 π_RL 生成的回答 y，奖励为 $ R(x, y) $。
3.  **算法**: 通常使用 **近端策略优化 (Proximal Policy Optimization, PPO)** 算法。PPO 是一种 Actor-Critic 方法，相对稳定且效率较高。
4.  **PPO 目标函数**: PPO 的目标是在最大化期望奖励的同时，限制 π_RL 相对于 π_SFT 的变化幅度。这是为了：
    *   **防止灾难性遗忘**: 避免模型忘记在 SFT 阶段学到的语言能力。
    *   **维持语言模型特性**: 确保生成的文本仍然流畅、相关。
    *   **探索与利用平衡**: 避免策略过于偏离初始策略，导致探索困难。
    *   **避免奖励模式过拟合 (Reward Hacking)**: 防止模型找到 RM 的漏洞来获得高分，但实际输出质量并不好。

    PPO 的目标函数（简化版，关注核心项）大致如下：

    $$
    Objective(φ) = E_{(x \in D_{prompt}, y\in π_{RL}(φ))} [ R(x, y) - β * KL( π_RL(φ)(y|x) || π_SFT(y|x) ) ]
    $$

    *   $ φ $: 当前 RL 策略 π_RL 的参数。
    *   $ D_prompt $: 用于 RL 训练的指令数据集。
    *   $ π_RL(φ)(y|x) $: 当前策略 π_RL（参数为 φ）在给定指令 x 时生成回答 y 的概率分布。
    *   $ π_SFT(y|x) $: 初始 SFT 模型生成回答 y 的概率分布（作为参考策略）。
    *   $ R(x, y) $: 奖励模型对 (x, y) 的打分。
    *   $ KL( π_RL || π_SFT ) $: π_RL 和 π_SFT 两个概率分布之间的 **KL 散度 (Kullback-Leibler divergence)**。它衡量了两个分布的差异程度。KL 散度项作为一个**惩罚项**，当 π_RL 与 π_SFT 差异过大时，惩罚会增加，阻止策略偏离太远。
    *   $ β $: KL 散度惩罚项的系数，控制惩罚的强度。这是一个需要调整的超参数。
    *   $ E $: 表示期望值，即对采样的指令和生成的回答计算平均目标值。

    **PPO 训练过程概要**:
    1.  **采样 (Rollout)**: 使用当前策略 π_RL 对一批指令 x 生成回答 y。
    2.  **评估 (Evaluation)**: 使用 RM 计算每个 (x, y) 对的奖励 R(x, y)。同时，计算 π_RL 和 π_SFT 生成该回答 y 的概率，用于计算 KL 散度惩罚。
    3.  **优势计算 (Advantage Estimation)**: PPO 使用一种叫做 Generalized Advantage Estimation (GAE) 的技术来估计在当前状态下采取某个动作（生成某个词）相对于平均水平的好坏程度。这通常需要一个价值函数（Value Function/Critic）的辅助，该价值函数也是在 RL 过程中学习的，用于预测给定状态（指令）下未来奖励的总和。
    4.  **策略更新 (Policy Update)**: 使用计算出的优势和 KL 散度惩罚，通过梯度上升更新策略 π_RL 的参数 φ，以最大化 PPO 目标函数。PPO 的核心机制（如 Clipped Surrogate Objective）确保每次更新步长不会太大，保证训练稳定性。

5.  **结果**: 得到最终的 RLHF 模型 (π_RL)，其生成的回答在符合人类偏好方面通常优于 SFT 模型。

**代码示例思路 (使用 Hugging Face TRL 库)**

直接编写 RLHF 的完整代码相当复杂，涉及 RL 算法的实现、模型并行、梯度同步等。幸运的是，Hugging Face 的 `trl` (Transformer Reinforcement Learning) 库极大地简化了这个过程。以下是使用 `trl` 实现 RLHF (特别是 PPO 阶段) 的概念性代码结构：

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, PPOConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

# ---------------- 配置参数 (Conceptual) ----------------
config = PPOConfig(
    model_name="path/to/your/sft_model",       # SFT 模型路径
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,                              # 每个 PPO 批次训练的 epoch 数
    kl_penalty="kl",                           # 使用 KL 散度惩罚
    target_kl=0.1,                             # 目标 KL 散度值 (如果使用 adaptive KL controller)
    init_kl_coef=0.2,                          # 初始 KL 系数 β (如果使用 fixed KL controller)
    adap_kl_ctrl=True,                         # 是否使用自适应 KL 控制器
    log_with="wandb",                          # 使用 wandb 记录日志 (可选)
    # ... 其他 PPO 相关参数
)

# ---------------- 1. 加载模型和 Tokenizer ----------------
# 加载 SFT 模型作为 PPO 的 Actor 和 Critic 的基础
# AutoModelForCausalLMWithValueHead 会在基础模型上添加一个价值头 (Value Head) 用于 PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# 加载奖励模型 (通常是另一个基于 Transformer 的序列分类模型)
reward_model = AutoModelForSequenceClassification.from_pretrained("path/to/your/reward_model")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token # 设置 pad token

# 创建参考模型 (Reference Model), 即 π_SFT, 用于计算 KL 散度
# 这通常是 SFT 模型的一个副本，在 PPO 训练中不更新参数
model_ref = create_reference_model(model)

# 将模型和奖励模型移动到合适的设备 (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
reward_model.to(device)
model_ref.to(device) # 参考模型也需要

# ---------------- 2. 准备数据集 ----------------
# 只需要包含指令 (prompt) 的数据集
# 假设数据集包含 'prompt' 列
dataset = load_dataset("your_prompt_dataset", split="train")

def tokenize_function(examples):
    # 对指令进行 tokenize
    return tokenizer(examples["prompt"], truncation=True, padding=False, max_length=128) # Max prompt length

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch") # 设置为 torch 格式

# ---------------- 3. 初始化 PPOTrainer ----------------
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
    dataset=tokenized_dataset,
    data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0]) # 简单的 collator
)

# ---------------- 4. PPO 训练循环 ----------------
generation_kwargs = {
    "min_length": -1, # 允许生成空字符串
    "top_k": 0.0,     # 不使用 top-k 采样
    "top_p": 1.0,     # 不使用 top-p (nucleus) 采样
    "do_sample": True,# 启用采样
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 64, # 控制生成回答的最大长度
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    # --- 步骤 1: 使用当前策略 π_RL 生成回答 (Rollout) ---
    # model 会同时计算 log probabilities (用于 PPO)
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch['response'] = tokenizer.batch_decode(response_tensors)

    # --- 步骤 2: 使用 RM 计算奖励 ---
    texts = [q + r for q, r in zip(batch['prompt'], batch['response'])] # 拼接 prompt 和 response
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    # 奖励模型输出 logits，需要取特定类别的分数或直接使用
    with torch.no_grad():
        raw_rewards = reward_model(**inputs).logits
        # 可能需要后处理 raw_rewards，例如取特定索引或进行归一化
        rewards = [torch.tensor(score) for score in raw_rewards] # 假设 reward model 输出单个分数

    # --- 步骤 3: 执行 PPO 优化步骤 ---
    # ppo_trainer.step 会计算 PPO 损失 (包括 KL 惩罚) 并更新 Actor (Policy) 和 Critic (Value Function)
    # 它需要 query tensors, response tensors, 和 rewards
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # 记录统计信息 (loss, rewards, KL divergence, etc.)
    ppo_trainer.log_stats(stats, batch, rewards)

# ---------------- 5. 保存模型 ----------------
ppo_trainer.save_pretrained("path/to/save/rlhf_model")
```

**关键点解释**:

*   **`AutoModelForCausalLMWithValueHead`**: 这是 `trl` 提供的关键类，它将一个标准的 Causal LM (如 GPT-2, LLaMA) 包装起来，并额外添加了一个线性层（Value Head），用于在 PPO 训练中学习价值函数 V(s)。
*   **`PPOTrainer`**: 封装了 PPO 算法的核心逻辑，包括生成、奖励计算接口、优势估计、损失计算和模型更新。
*   **`create_reference_model`**: 创建一个与主模型结构相同但不参与训练的参考模型，用于计算 KL 散度惩罚项 `KL(π_RL || π_SFT)`。
*   **奖励计算**: 在 PPO 循环中，需要手动调用奖励模型来为生成的 `(prompt, response)` 对打分。这个分数 `rewards` 会被传入 `ppo_trainer.step`。
*   **`ppo_trainer.step`**: 这是执行 PPO 更新的核心函数。它接收采样的 prompts, responses, 和对应的 rewards，然后在内部完成 PPO 的复杂计算（优势估计、计算策略损失和价值损失、KL 散度惩罚、梯度更新等）。
*   **KL 散度控制 (`beta` / `kl_penalty`)**: `PPOConfig` 中的 `kl_penalty`, `target_kl`, `init_kl_coef`, `adap_kl_ctrl` 等参数控制了 PPO 目标函数中 KL 散度惩罚项的行为，这是维持模型稳定性和防止奖励模式过拟合的关键。

**总结**

RLHF 是一个强大的技术，通过结合人类反馈和强化学习，能够显著提升 LLM 与人类意图和偏好的一致性，使其更安全、更有用。它是一个多阶段、相对复杂的过程，涉及数据收集、奖励模型训练和 PPO 优化，但像 `trl` 这样的库使得实践 RLHF 变得更加可行。理解其背后的原理、流程和关键公式对于应用和研究 LLM alignment至关重要。

## DPO：二分类器

Direct Preference Optimization (DPO) 是一种直接利用人类偏好数据优化语言模型的方法，避免了传统强化学习从人类反馈（RLHF）中复杂的奖励模型训练和策略优化步骤。以下从公式、作用及代码实现三方面详细说明：

---

### **1. DPO 的核心思想与作用**
**目标**：直接通过偏好数据（即对于同一提示，人类标注的优选和劣选回答）调整模型，使其生成更符合人类偏好的输出，同时避免过度偏离原始模型（保持稳定性）。

**优势**：
- **简化流程**：省去奖励模型训练和强化学习（如PPO）步骤。
- **稳定性**：监督式训练更稳定，避免RL的不稳定性。
- **效率**：计算成本低，易于实现。

---

### **2. 数学公式推导**
#### **传统RLHF的优化目标**
RLHF通过最大化奖励并约束KL散度：
\[
\max_{\pi} \mathbb{E}_{x,y \sim \mathcal{D}} \left[ r_{\phi}(x,y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi(y|x) \| \pi_{\text{ref}}(y|x) \right]
\]
其中 \(r_{\phi}\) 是奖励模型，\(\beta\) 是KL惩罚系数。

#### **DPO的重参数化**
DPO将奖励 \(r(x,y)\) 隐式表示为策略模型 \(\pi\) 和参考模型 \(\pi_{\text{ref}}\) 的对数概率差：
\[
r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
\]
代入Bradley-Terry偏好模型：
\[
p(y_w \succ y_l | x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}
\]
得到DPO损失函数：
\[
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\]
其中 \(\sigma\) 是sigmoid函数，\(y_w\) 和 \(y_l\) 分别为优选和劣选回答。

---

### **3. 代码实现详解**
#### **关键步骤**
1. **计算对数概率**：对优选和劣选回答，分别用策略模型 $\pi(y_w|x)$ 和参考模型 $\pi_{ref}(y_w|x)$ 计算序列的对数概率。
2. **损失计算**：按DPO公式计算损失并反向传播。

#### **示例代码（PyTorch框架）**
```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_model, ref_model, beta, x, y_w, y_l):
    ref_model.eval()
    
    # 获取策略模型的对数概率
    pi_logits_w = pi_model(x, y_w).log_softmax(dim=-1)
    pi_logprob_w = pi_logits_w.sum(dim=-1)  # 序列对数概率求和
    pi_logits_l = pi_model(x, y_l).log_softmax(dim=-1)
    pi_logprob_l = pi_logits_l.sum(dim=-1)
    
    # 获取参考模型的对数概率（detach梯度）
    with torch.no_grad():
        ref_logits_w = ref_model(x, y_w).log_softmax(dim=-1)
        ref_logprob_w = ref_logits_w.sum(dim=-1)
        ref_logits_l = ref_model(x, y_l).log_softmax(dim=-1)
        ref_logprob_l = ref_logits_l.sum(dim=-1)
    
    # 计算对数比值差
    ratio_w = pi_logprob_w - ref_logprob_w
    ratio_l = pi_logprob_l - ref_logprob_l
    diff = beta * (ratio_w - ratio_l)
    
    # 计算损失
    loss = -F.logsigmoid(diff).mean()
    return loss
```

#### **代码说明**
- **输入**：策略模型 `pi_model`，固定参考模型 `ref_model`，参数 `beta`，输入提示 `x`，优选回答 `y_w`，劣选回答 `y_l`。
- **对数概率计算**：使用 `log_softmax` 获取每个token的对数概率，求和得整个序列的对数概率。
- **参考模型处理**：通过 `with torch.no_grad()` 阻止梯度传播。
- **损失计算**：按公式计算差异 `diff`，并应用 `logsigmoid` 得到损失。

---

### **4. 超参数与注意事项**
- **β的选择**：控制模型偏离参考模型的程度，通常需调参（例如0.1-0.5）。
- **数据格式**：偏好数据需为三元组 `(x, y_w, y_l)`。
- **模型对齐**：确保 `pi_model` 和 `ref_model` 的tokenizer和模型结构一致。

---

### **总结**
DPO通过隐式奖励建模将强化学习问题转化为监督学习，直接优化模型生成符合人类偏好的输出。其核心在于利用策略模型与参考模型的对数概率差替代显式奖励，简化训练流程并提升稳定性。代码实现需注意高效计算序列对数概率并正确处理参考模型的梯度。

## 使用DPO优化SonnetGPT

要用 DPO（Direct Preference Optimization）优化你的 SonnetGPT，需要以下步骤：

### 1. 数据准备

DPO 需要**成对的样本**，每对包含：
- **Prompt**（输入，如前三行诗）
- **Preferred Output**（更好的生成结果）
- **Less Preferred Output**（较差的生成结果）

可以这样收集数据：
- 用当前模型生成多个候选诗歌结尾，让人工标注哪一个更好。
- 或者用已有的高质量诗歌作为“preferred”，用模型生成的普通输出作为“less preferred”。

数据格式建议如下（JSONL，每行一个样本）：
```json
{
  "prompt": "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,",
  "chosen": "And summer’s lease hath all too short a date:\nSometime too hot the eye of heaven shines,\nAnd often is his gold complexion dimm’d;",
  "rejected": "The sun is bright and the sky is blue,\nFlowers bloom and the grass is new,\nBirds sing sweetly in the morning dew,"
}
```

---

### 2. 代码修改建议

#### (1) 新建 DPO 数据集类
```python
import torch
from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        prompt_ids = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length)
        chosen_ids = self.tokenizer(chosen, return_tensors='pt', truncation=True, max_length=self.max_length)
        rejected_ids = self.tokenizer(rejected, return_tensors='pt', truncation=True, max_length=self.max_length)

        return {
            'prompt_ids': prompt_ids['input_ids'].squeeze(0),
            'chosen_ids': chosen_ids['input_ids'].squeeze(0),
            'rejected_ids': rejected_ids['input_ids'].squeeze(0)
        }

    def __len__(self):
        return len(self.data)
```

#### (2) 实现 DPO 损失
DPO 损失核心思想：让模型对“chosen”输出的概率高于“rejected”。
```python
import torch.nn.functional as F

def dpo_loss(model, batch, beta=0.1):
    # 获取 logits
    prompt_ids = batch['prompt_ids'].to(model.get_device())
    chosen_ids = batch['chosen_ids'].to(model.get_device())
    rejected_ids = batch['rejected_ids'].to(model.get_device())

    # 拼接 prompt+output
    chosen_input = torch.cat([prompt_ids, chosen_ids], dim=1)
    rejected_input = torch.cat([prompt_ids, rejected_ids], dim=1)
    attn_chosen = torch.ones_like(chosen_input)
    attn_rejected = torch.ones_like(rejected_input)

    # 计算 log_probs
    logits_chosen = model(chosen_input, attn_chosen)
    logits_rejected = model(rejected_input, attn_rejected)

    # 只计算 output 部分的 log_prob
    chosen_logprob = F.cross_entropy(
        logits_chosen[:, -chosen_ids.size(1)-1:-1, :].reshape(-1, logits_chosen.size(-1)),
        chosen_ids.flatten(),
        reduction='none'
    ).reshape(chosen_ids.shape).sum(dim=1)

    rejected_logprob = F.cross_entropy(
        logits_rejected[:, -rejected_ids.size(1)-1:-1, :].reshape(-1, logits_rejected.size(-1)),
        rejected_ids.flatten(),
        reduction='none'
    ).reshape(rejected_ids.shape).sum(dim=1)

    # DPO 损失
    loss = -torch.log(torch.sigmoid(beta * (rejected_logprob - chosen_logprob))).mean()
    return loss
```

#### (3) 训练循环
- 用 `DPODataset` 加载数据
- 用 `dpo_loss` 替换原有 loss 计算

---

### 3. 总结

- **数据**：prompt + preferred output + less preferred output
- **代码**：实现 DPO 数据集、DPO 损失，替换训练循环
- **训练**：用人工偏好数据微调模型