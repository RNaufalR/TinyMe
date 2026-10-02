# Tool-use evaluation — EXP-004-TOOL-SFT-V3 (best)

- Suite: independent cases A-H, executed through the real agent runtime + sandbox
- Dataset: `dataset_v3` · temperature 0.0 · greedy decoding
- Generated: 2026-10-02T12:41:53 (3.64 s)

| Metric | fp32 |
| :--- | ---: |
| tool_needed_accuracy | 0.3000 |
| tool_not_needed_accuracy | 0.4000 |
| tool_syntax_validity | 0.8400 |
| tool_name_accuracy | 0.3000 |
| argument_validity | 1.0000 |
| argument_accuracy | 0.6000 |
| execution_success | 0.5000 |
| multi_step_success | 0.0000 |
| error_recovery_success | 0.0000 |
| grounded_final_answer | 0.3000 |
| citation_validity | 0.0000 |
| task_completion | 0.0000 |

## Per-family task completion

| Family | fp32 |
| :--- | ---: |
| A | 0.0000 |
| B | 0.0000 |
| C | 0.0000 |
| D | 0.0000 |
| E | 0.0000 |
| F | 0.0000 |
| G | 0.0000 |

## Example trajectories (first case per family, fp32)

**A0** — 'What is 21 + 34?'  
stop=`no_final_no_tool_call` tools=['compute'] final=``

**B0** — 'Compute exactly: 4837 * 962 + 71'  
stop=`no_final_no_tool_call` tools=['compute'] final=``

**C0** — 'What is the population of jakarta according to a source?'  
stop=`no_final_no_tool_call` tools=['compute'] final=``

**D0** — 'Read the stored source FACT-988002 about the population of jakarta and quote its key sentence.'  
stop=`protocol_error:call_not_json` tools=[] final=``

**E0** — 'Run this program in the sandbox and report the result:\n```python\ndef is_prime(n):\n    if n < 2:\n        return'  
stop=`final` tools=[] final=`{"
The matching index is a`

**F0** — 'The tests fail for this code:\n```python\ndef count_words(text):\n    counts = {}\n    for w in text.lower().split'  
stop=`final` tools=[] final=`def sum_words(text, x) that returns a list of the documented type.`

**G0** — 'Look up the population of jakartaa (note the typo) and if the search finds nothing, correct the query and answ'  
stop=`final` tools=[] final=`Plan the operations in order and give the result.
<|final|>
def value, right implement x: Write a function running_max(set) that returns a list of nesting in a `

