# RLVR-Wordle Project Notes

## Project Goal

Compare **sparse vs dense reward signals** for teaching an LLM to play Wordle using RL + LoRA.

| Condition | Reward Signal |
|-----------|---------------|
| Sparse | One reward at end (win/lose, or guess count) |
| Dense | Reward after each guess (greens, yellows, information gain, etc.) |

**Hypothesis:** Dense should learn faster (more signal), but sparse might generalize better (less reward hacking risk).

---

## Hardware & Constraints

- **Machine:** MacBook Pro M1 Pro
- **Memory:** Assume 16GB unified memory
- **Backend:** PyTorch MPS (not CUDA)
- **Limitation:** TRL/GRPO are CUDA-centric — will need custom training loop or Colab

---

## Model Choice

**Qwen3-1.7B** — tested and can play Wordle out of the box (0.6B couldn't).

Memory estimate for training with LoRA:
- Model (fp16): ~3.5GB
- LoRA adapters: ~50-100MB
- Optimizer states: ~200-400MB
- Activations: ~2-4GB
- **Total:** ~6-8GB ✓

---

## Algorithm Choice

### REINFORCE vs GRPO

Both are policy gradient methods. GRPO = REINFORCE with a specific baseline (group mean).

| | REINFORCE | GRPO |
|---|-----------|------|
| Samples per prompt | 1 | K (e.g., 8-64) |
| Baseline | None or global | Per-prompt group mean |
| Value network | No | No |
| Complexity | Simpler | Slightly more complex |

**For this experiment:** Use REINFORCE with reward-to-go for both conditions. This isolates the reward signal as the only variable.

### Why Not PPO?

Overkill. Recent research (Schulman's "LoRA Without Regret") shows simpler algorithms work just as well for LLM fine-tuning because:
- LLMs start from strong pretrained weights
- Lower variance than training from scratch
- No need for value network complexity

---

## LoRA Configuration

### Key Insight from Schulman Paper

> "LoRA fully matches the learning performance of FullFT for RL, even with ranks as low as 1."

RL needs very low capacity because policy gradients provide ~1 bit of information per episode.

### Recommended Settings

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,  # rank (could go as low as 4 for RL)
    lora_alpha=32,  # scaling factor
    target_modules=[
        # Apply to ALL layers, not just attention
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,  # rank-stabilized LoRA — keeps gradient magnitude consistent across ranks
)
```

### Why These Choices?

1. **Apply to all layers (especially MLP):** Attention-only underperforms even with matched parameter count
2. **rsLoRA:** Standard LoRA scales by α/r, making higher ranks learn slower. rsLoRA uses α/√r to normalize gradient magnitudes
3. **Learning rate:** ~10x higher than full fine-tuning (e.g., 1e-4 instead of 1e-5)

---

## Reward Design

### Sparse Reward (Condition A)

```python
def sparse_reward(won: bool, num_guesses: int) -> float:
    if won:
        return 1.0  # or: 7 - num_guesses for guess-count bonus
    else:
        return 0.0  # or: -1.0 for penalty
```

### Dense Reward (Condition B)

```python
def dense_step_reward(guess: str, feedback: str, game_state: GameState) -> float:
    r = 0.0
    
    # Reward for feedback quality
    r += feedback.count('🟩') * 0.4  # greens
    r += feedback.count('🟨') * 0.2  # yellows
    
    # Reward for information gain (narrowing solution space)
    words_before = len(game_state.possible_words)
    words_after = len(filter_words(game_state.possible_words, guess, feedback))
    r += (words_before - words_after) / words_before * 0.3
    
    # Penalty for ignoring previous feedback
    if uses_known_gray_letter(guess, game_state):
        r -= 0.5
    if ignores_known_green(guess, game_state):
        r -= 0.5
    
    return r
```

### Combining Dense + Final Bonus

To prevent reward hacking (maximizing intermediate rewards without winning):

```python
def total_episode_reward(step_rewards: list, won: bool) -> float:
    dense_sum = sum(step_rewards)
    final_bonus = 10.0 if won else -5.0
    return dense_sum + final_bonus
```

---

## Training Loop (REINFORCE)

### Core Algorithm

```python
def compute_loss_sparse(episode):
    """Sparse: single reward at end"""
    final_reward = 1.0 if episode.won else 0.0
    total_log_prob = sum(episode.log_probs)  # sum over all guesses
    return -total_log_prob * final_reward

def compute_loss_dense(episode):
    """Dense: per-step reward with reward-to-go"""
    loss = 0.0
    rewards = episode.step_rewards
    
    for t in range(len(episode.guesses)):
        reward_to_go = sum(rewards[t:])  # R_t + R_{t+1} + ... + R_T
        loss -= episode.log_probs[t] * reward_to_go
    
    return loss
```

### Full Training Loop Skeleton

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# 1. Load model with LoRA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = get_peft_model(model, lora_config)
model.to("mps")  # Apple Silicon

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. Training loop
for episode_num in range(num_episodes):
    # Play one game of Wordle
    episode = play_wordle(model, tokenizer, target_word)
    
    # Compute loss (sparse or dense depending on condition)
    loss = compute_loss(episode)
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging
    if episode_num % 100 == 0:
        print(f"Episode {episode_num}, Win rate: {compute_win_rate()}")
```

### Getting Log Probabilities

The key LLM-specific part — computing log π(token | context):

```python
def get_log_probs(model, tokenizer, prompt: str, completion: str):
    """Get log probabilities for each token in completion"""
    
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer(prompt).input_ids)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get log probs for completion tokens only
    log_probs = []
    for i in range(prompt_len, len(inputs.input_ids[0])):
        token_id = inputs.input_ids[0, i]
        token_logits = logits[0, i - 1]  # logits for predicting this token
        token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
        log_probs.append(token_log_prob)
    
    return torch.stack(log_probs)
```

---

## Wordle Environment

### Core Game Logic

```python
class WordleGame:
    def __init__(self, target_word: str, valid_words: list[str]):
        self.target = target_word.upper()
        self.valid_words = set(w.upper() for w in valid_words)
        self.guesses = []
        self.feedbacks = []
        self.max_guesses = 6
    
    def guess(self, word: str) -> tuple[str, bool]:
        """Returns (feedback_string, game_over)"""
        word = word.upper()
        
        if len(word) != 5 or word not in self.valid_words:
            return None, False  # invalid guess
        
        feedback = self._compute_feedback(word)
        self.guesses.append(word)
        self.feedbacks.append(feedback)
        
        won = word == self.target
        game_over = won or len(self.guesses) >= self.max_guesses
        
        return feedback, game_over
    
    def _compute_feedback(self, guess: str) -> str:
        result = ['⬜'] * 5
        target_chars = list(self.target)
        
        # First pass: mark greens
        for i, (g, t) in enumerate(zip(guess, self.target)):
            if g == t:
                result[i] = '🟩'
                target_chars[i] = None
        
        # Second pass: mark yellows
        for i, g in enumerate(guess):
            if result[i] == '⬜' and g in target_chars:
                result[i] = '🟨'
                target_chars[target_chars.index(g)] = None
        
        return ''.join(result)
```

### Prompting the LLM

```python
def format_prompt(game: WordleGame) -> str:
    prompt = """You are playing Wordle. Guess a 5-letter English word.

Rules:
- 🟩 = correct letter, correct position
- 🟨 = correct letter, wrong position
- ⬜ = letter not in word

"""
    if game.guesses:
        prompt += "Previous guesses:\n"
        for guess, feedback in zip(game.guesses, game.feedbacks):
            prompt += f"{guess}: {feedback}\n"
        prompt += "\n"
    
    prompt += "Your guess:"
    return prompt
```

---

## Expected Training Scale

### Speed Estimates (M1 Pro)

| Operation | Time |
|-----------|------|
| Single forward pass | ~0.5-1 sec |
| Generate one guess (~20-50 tokens) | ~2-5 sec |
| Full episode (up to 6 guesses) | ~15-30 sec |
| One training step | ~1-2 sec |
| **One full iteration** | ~20-40 sec |

### How Many Episodes?

| Milestone | Episodes | Time |
|-----------|----------|------|
| See any learning signal | 100-300 | ~1-2 hours |
| Noticeable improvement | 500-2,000 | ~4-15 hours |
| Actually good | 3,000-10,000+ | ~1-4 days |

Start with 200 episodes and evaluate. Don't plan for 10,000 upfront.

---

## Project Structure

```
rlvr-wordle/
├── README.md
├── requirements.txt
├── src/
│   ├── environment.py    # WordleGame class, reward functions
│   ├── model.py          # Load model + LoRA setup
│   ├── train.py          # REINFORCE training loop
│   └── evaluate.py       # Test win rate, compare conditions
├── configs/
│   ├── sparse.yaml       # Sparse reward config
│   └── dense.yaml        # Dense reward config
├── notebooks/
│   └── exploration.ipynb # Quick experiments
└── results/
    └── ...               # Logs, checkpoints, plots
```

---

## Key Resources

### Must Read
1. [Cameron Wolfe: REINFORCE for LLMs](https://cameronrwolfe.substack.com/p/reinforce) — LLM-specific RL, why simple algorithms work
2. [Schulman: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — LoRA matches full fine-tuning for RL, even rank 1
3. [Kalomaze: RL + LoRA Deep Dive](https://kalomaze.bearblog.dev/rl-lora-ddd/) — Practical LoRA rank selection for RL

### Reference
4. [HuggingFace PEFT LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) — Config options
5. [Janu Verma: RL Fine-tuning from Scratch](https://januverma.substack.com/p/fine-tuning-llms-using-reinforcement) — Clean PyTorch implementation
6. [TRL Wordle GRPO Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) — Reference (but build your own)

---

## Open Questions / Decisions

1. **Exact dense reward formula** — what weights for greens/yellows/information gain?
2. **Baseline for REINFORCE** — running average? per-prompt? none?
3. **Word list** — use official Wordle list or broader English 5-letter words?
4. **Evaluation metric** — win rate? average guesses? both?
5. **How many seeds** — for statistical significance on sparse vs dense comparison

---

## Quick Start Checklist

- [ ] Set up repo structure
- [ ] Implement WordleGame environment
- [ ] Test Qwen3-1.7B can generate valid guesses (no training)
- [ ] Implement reward functions (sparse + dense)
- [ ] Implement log probability extraction
- [ ] Implement REINFORCE training loop
- [ ] Run 100 episodes sparse, check for any signal
- [ ] Run 100 episodes dense, compare
- [ ] Iterate on reward design if needed
- [ ] Full experiment: N episodes × 2 conditions × M seeds
- [ ] Analyze results, write up findings