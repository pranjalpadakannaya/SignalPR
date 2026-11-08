# prompts/ — versioned prompts (issue #3)

Every prompt the agent sends to the model will live here as a text file, one folder per step:

```
prompts/
  intent/v1.txt
  specialist_correctness/v1.txt
  specialist_security/v1.txt
  specialist_design/v1.txt
  judge/v1.txt
```

Rules:

- Never edit a version that has been used in a logged run. Copy it to `v2.txt` and change the copy.
- Log the version you used: `python -m tracking log ... --prompt-version v2`.
- Tune prompt versions on Martian (dev) only.

The prompts are still inline strings in `agent/` until the agent pipeline (#19) is merged; they will
be moved here after that, starting as `v1`.
