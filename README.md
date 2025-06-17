# 🤖 LLM-Conversations: Personality-Driven AI Dialogue

This project demonstrates a fun and experimental interaction between two LLM-powered agents — one **rude and sarcastic**, the other **kind and constructive** — as they debate or discuss a given topic. It explores the idea of injecting **personality, behavior control**, and **back-and-forth dialogue** into large language models.

---

## 📌 Project Goals

- Simulate multi-agent conversation between two contrasting personalities.
- Use **structured system prompts** to influence model behavior.
- Test the **limits of instruction-following** in local LLMs (like Qwen or Gemma).
- Explore how models maintain context and opposition over multiple turns.

---

## 🧠 How It Works

- Two conversational agents are defined using structured system prompts:
  - 🗯️ `RudeBot`: Sarcastic, blunt, negative tone. Responds with a max of 20–30 words.
  - 🌸 `KindBot`: Polite, factual, positive tone. Also limited to 20 words.

- Agents are implemented using locally hosted models (e.g., `Qwen/Qwen3-4B` via Hugging Face or Ollama).

- Each model receives the other’s reply and responds accordingly, maintaining their personalities while engaging in dialogue.

---

## 🛠️ Tech Stack

- **Python**
- **Transformers** (Hugging Face)
- **PyTorch**
- **Local LLMs** (tested on Qwen3, Gemma via `AutoModelForCausalLM`)
- **Ollama** (optional)
- **Jupyter Notebook** for experiment orchestration

---

## 📂 Example Output

- User Input: Will AI replace human jobs?

- **RudeBot**: Of course it will. You're delusional if you think your job matters in the future.

- **KindBot**: That’s unlikely. AI will assist, not replace, most roles—especially creative or human-centric ones.
