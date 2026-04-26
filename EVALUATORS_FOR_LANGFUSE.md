# Suggested Langfuse evaluators for HT Lektion 12

Create at least these two evaluators in Langfuse UI.

---

## Evaluator 1: answer_relevance
- **Type:** numeric
- **Scale:** 0 to 1

### Prompt
Judge whether the assistant output answers the user's request.

User input:
{{input}}

Assistant output:
{{output}}

Scoring guide:
- 1.0 = directly answers the request, stays on topic, and is useful
- 0.5 = partially answers but misses important parts
- 0.0 = irrelevant or misleading

Return only the numeric score.

---

## Evaluator 2: groundedness_check
- **Type:** boolean

### Prompt
Decide whether the assistant output appears grounded in retrieved evidence and avoids unsupported claims.

User input:
{{input}}

Assistant output:
{{output}}

Return:
- true = mostly grounded and evidence-based
- false = contains obvious unsupported claims or speculation

---

## Optional 3rd evaluator: structure_quality
- **Type:** categorical
- **Labels:** poor, acceptable, strong

### Prompt
Rate the structure and readability of the assistant output.

User input:
{{input}}

Assistant output:
{{output}}

Criteria:
- logical order
- concise sections
- readable formatting
- clear sources if relevant

Return only one label: poor, acceptable, or strong.


---

## Critic_Feedback_adoption
- **Type:** numeric

### Prompt
Evaluate whether the final assistant response properly addresses the critic's revision feedback.

User request:
{{input}}

Final assistant response:
{{output}}

Judge the response by these criteria:
1. Does the final response appear improved and complete?
2. Does it address likely missing points or weaknesses that a critic would flag?
3. Does it avoid obvious omissions, vagueness, or ignored issues?
4. Is the answer now more actionable, structured, and useful?

Return:
- true if the final response appears to have incorporated critique well
- false if major likely critique points still seem ignored
