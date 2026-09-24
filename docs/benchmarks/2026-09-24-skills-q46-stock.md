# Jetson skill-routing measurement -- stock, 2026-09-24

## Run

- command: `measure_skills.py --tools $HOME/lfm-train/work/q46/skills/tools.json --test $HOME/lfm-train/work/q46/skills/test.jsonl --manifest $HOME/lfm-train/work/q46/skills/manifest.json --url <local endpoint> --model stock --model-revision 2fc06364715b967f1860aea9cf38778875588b17 --label stock --enable-thinking false --timeout 180 --out $HOME/lfm-train/work/q46/measure/skills-stock.md`
- date: 2026-09-24
- endpoint: `<local endpoint>` (a local endpoint was used; never recorded)
- model: `stock`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- thinking: chat_template_kwargs enable_thinking=false
- device: <https://github.com/NVIDIA-AI-IOT/jetson-device-skills> at commit `20137897aef549cc2fa36a18c10e45da94967c3e`
- bsp: <https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills> at commit `fdfafef0416be1eb3852b68fc802a9752f33fb28`

## Results

| Split | Correct / total |
|---|---|
| overall | 42 of 104 (40%) |
| skill named in prompt | 14 of 34 (41%) |
| skill not named | 28 of 70 (40%) |
| repo: bsp | 16 of 48 (33%) |
| repo: device | 26 of 56 (46%) |

| Outcome | Count |
|---|---|
| correct | 42 |
| wrong_skill | 27 |
| no_call | 31 |
| several_calls | 4 |
| call_error | 0 |
| Non-empty think blocks (must be 0) | 0 |

| Latency | ms |
|---|---|
| median | 206 |
| p95 | 5110 |
