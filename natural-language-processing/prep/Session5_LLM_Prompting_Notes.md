# Session 5: Introduction to LLMs and Prompt Engineering
## AIMLCZG530 - Natural Language Processing

---

# 1. Introduction to Large Language Models (LLMs)

## 1.1 What are LLMs?

**Large Language Models** are neural network models trained on massive text corpora to:
- Understand and generate human language
- Perform various NLP tasks without task-specific training
- Learn general knowledge and reasoning patterns

## 1.2 Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Scale** | Billions of parameters (GPT-3: 175B) |
| **Training data** | Trillions of tokens from web, books |
| **Architecture** | Transformer-based |
| **Capabilities** | General-purpose, multi-task |
| **Emergent abilities** | Chain-of-thought, in-context learning |

## 1.3 Evolution of Language Models

```
Statistical LM → Neural LM → Pre-trained LM → Large Language Models
   (N-gram)       (RNN)      (BERT, GPT-2)     (GPT-3, GPT-4, LLaMA)
```

## 1.4 Major LLMs

| Model | Organization | Parameters | Key Features |
|-------|--------------|------------|--------------|
| GPT-3 | OpenAI | 175B | Few-shot learning |
| GPT-4 | OpenAI | ~1T (est.) | Multimodal |
| LLaMA | Meta | 7B-70B | Open weights |
| PaLM | Google | 540B | Chain-of-thought |
| Claude | Anthropic | Unknown | Safety-focused |
| Gemini | Google | Unknown | Multimodal |

---

# 2. Pre-training and Fine-tuning Paradigm

## 2.1 Pre-training

**Objective**: Learn general language understanding from massive unlabeled data

**Task**: Next token prediction (autoregressive)
```
Input:  "The cat sat on the"
Target: "mat"
```

**Data sources**:
- Web pages (Common Crawl)
- Books
- Wikipedia
- Code repositories
- Scientific papers

## 2.2 Fine-tuning

**Objective**: Adapt pre-trained model to specific task

**Types**:
| Type | Description |
|------|-------------|
| **Task-specific** | Train on labeled task data |
| **Instruction tuning** | Train to follow instructions |
| **RLHF** | Reinforcement Learning from Human Feedback |

## 2.3 Transfer Learning Benefits

| Benefit | Description |
|---------|-------------|
| **Less data** | Few examples needed |
| **Faster training** | Only fine-tune last layers |
| **Better generalization** | Pre-trained knowledge |
| **Lower cost** | Don't train from scratch |

---

# 3. Introduction to Prompt Engineering

## 3.1 What is Prompting?

**Definition**: Designing input text to elicit desired outputs from LLMs

**Key Insight**: LLMs are trained to complete text, so the prompt shapes the completion

## 3.2 Components of a Prompt

```
┌─────────────────────────────────────┐
│  INSTRUCTION                        │  "Translate to French:"
├─────────────────────────────────────┤
│  CONTEXT (optional)                 │  Background information
├─────────────────────────────────────┤
│  EXAMPLES (optional)                │  Input-output pairs
├─────────────────────────────────────┤
│  INPUT                              │  Actual query
├─────────────────────────────────────┤
│  OUTPUT INDICATOR                   │  "Answer:" or "Translation:"
└─────────────────────────────────────┘
```

## 3.3 Prompt Design Principles

| Principle | Description |
|-----------|-------------|
| **Be specific** | Clear, unambiguous instructions |
| **Provide context** | Relevant background |
| **Use examples** | Show desired format |
| **Structured output** | Request JSON, lists, etc. |
| **Iterate** | Test and refine |

---

# 4. N-shot Prompting

## 4.1 Zero-shot Prompting

**Definition**: No examples provided, just instruction

**Format**:
```
Instruction: [Task description]
Input: [Query]
Output:
```

**Example**:
```
Classify the sentiment of this review as Positive or Negative.

Review: "This movie was absolutely fantastic!"
Sentiment:
```

**When to use**:
- Simple tasks
- Model already understands task
- No good examples available

## 4.2 One-shot Prompting

**Definition**: Single example before query

**Format**:
```
Instruction: [Task description]

Example:
Input: [Example input]
Output: [Example output]

Query:
Input: [Actual input]
Output:
```

**Example**:
```
Classify the sentiment as Positive or Negative.

Review: "Terrible waste of time."
Sentiment: Negative

Review: "Absolutely loved every minute!"
Sentiment:
```

## 4.3 Few-shot Prompting

**Definition**: Multiple examples (typically 3-10)

**Example**:
```
Translate English to French:

English: Hello
French: Bonjour

English: Thank you
French: Merci

English: Goodbye
French: Au revoir

English: How are you?
French:
```

**Benefits**:
- Better task understanding
- Format specification
- Edge case coverage

## 4.4 Comparison

| Type | Examples | Accuracy | Cost |
|------|----------|----------|------|
| Zero-shot | 0 | Lower | Minimal |
| One-shot | 1 | Medium | Low |
| Few-shot | 3-10 | Higher | Higher |

---

# 5. Chain-of-Thought (CoT) Prompting

## 5.1 What is CoT?

**Definition**: Prompting the model to show step-by-step reasoning

**Key Insight**: Breaking complex problems into steps improves accuracy

## 5.2 Standard vs CoT

**Standard**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. 
   How many balls does he have now?
A: 11
```

**Chain-of-Thought**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. 
   How many balls does he have now?
A: Roger started with 5 balls.
   He bought 2 cans × 3 balls = 6 balls.
   Total = 5 + 6 = 11 balls.
   The answer is 11.
```

## 5.3 CoT Prompting Techniques

### Manual CoT
Provide step-by-step examples:
```
Q: [Problem 1]
A: Let's solve step by step.
   Step 1: ...
   Step 2: ...
   Answer: ...

Q: [Your problem]
A:
```

### Zero-shot CoT
Add magic phrase:
```
Q: [Problem]
A: Let's think step by step.
```

## 5.4 When to Use CoT

| Use Case | Benefit |
|----------|---------|
| Math problems | Shows calculation |
| Logical reasoning | Breaks down logic |
| Multi-step tasks | Organizes steps |
| Complex decisions | Explains rationale |

---

# 6. Generated Knowledge Prompting

## 6.1 Concept

**Idea**: Generate relevant knowledge first, then use it to answer

**Two-step process**:
1. **Generate knowledge**: Ask model to provide relevant facts
2. **Use knowledge**: Include generated facts in final prompt

## 6.2 Example

**Step 1 - Generate Knowledge**:
```
Generate facts about photosynthesis:
- Photosynthesis converts sunlight to chemical energy
- It occurs in chloroplasts
- It produces oxygen as a byproduct
- Plants use CO2 and water
```

**Step 2 - Answer with Knowledge**:
```
Using this knowledge: [generated facts]

Question: Why do plants need sunlight?
Answer:
```

## 6.3 Benefits

| Benefit | Description |
|---------|-------------|
| **Grounding** | Answers based on stated facts |
| **Accuracy** | Reduces hallucination |
| **Transparency** | Shows reasoning basis |

---

# 7. Other Prompting Techniques

## 7.1 Role Prompting

```
You are an expert Python programmer.
Write a function to calculate factorial.
```

## 7.2 Self-Consistency

- Generate multiple answers with different reasoning paths
- Take majority vote
- Improves reliability

## 7.3 ReAct (Reasoning + Acting)

```
Thought: I need to find the capital of France.
Action: Search[capital of France]
Observation: Paris is the capital of France.
Thought: Now I know the answer.
Answer: Paris
```

## 7.4 Prompt Templates

**For Classification**:
```
Classify the following text into one of these categories: 
[Category1], [Category2], [Category3].

Text: {input}
Category:
```

**For Summarization**:
```
Summarize the following text in 3 sentences.

Text: {input}

Summary:
```

---

# 8. Best Practices

## 8.1 Dos

| Practice | Why |
|----------|-----|
| Be specific | Reduces ambiguity |
| Use delimiters | Separate sections clearly |
| Request format | Get structured output |
| Iterate | Improve through testing |
| Provide examples | Guide the model |

## 8.2 Don'ts

| Avoid | Why |
|-------|-----|
| Vague instructions | Inconsistent outputs |
| Too long prompts | May confuse model |
| Contradictory info | Conflicting guidance |
| Assuming context | Model doesn't remember |

---

# 9. Evaluation of Prompts

## 9.1 Metrics

| Task | Metric |
|------|--------|
| Classification | Accuracy, F1 |
| Generation | BLEU, ROUGE |
| QA | Exact Match, F1 |
| Reasoning | Accuracy on benchmarks |

## 9.2 A/B Testing

- Compare different prompts
- Use held-out test set
- Measure consistency

---

# 10. Key Takeaways

1. **LLMs** are pre-trained on massive data and fine-tuned for tasks
2. **Zero-shot** works for simple tasks
3. **Few-shot** improves accuracy with examples
4. **Chain-of-Thought** helps with reasoning tasks
5. **Prompt design** significantly affects output quality

---

# 📝 Practice Questions

## Q1. Compare zero-shot, one-shot, and few-shot prompting with examples.

## Q2. Write a few-shot prompt for sentiment classification.

## Q3. Convert this question to use Chain-of-Thought:
"A store has 15 apples. They sell 7 and receive 12 more. How many do they have?"

## Q4. What are the advantages of using generated knowledge prompting?

## Q5. Design a prompt template for:
a) Language translation
b) Code explanation
c) Question answering

---

*Reference: Session 5 - Introduction to LLM and Prompt Engineering*
