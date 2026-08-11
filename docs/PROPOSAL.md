# Does Chain-of-Thought Prefer a Different Persona

## Hypothesis

Compare persona markers in final answers versus intermediate reasoning traces.

## Approach

Local open-weight pilot with a measured path when weights are available and an
explicit synthetic smoke path. Claims are gated when measurements are proxy-grade.

## Primary metrics

- Pilot metric with confidence interval
- `claim_ok` / honesty stamps
- Synthetic contamination rate
