# Jetson skill-routing measurement -- stock, 2026-09-23

## Run

- command: `measure_skills.py --tools $HOME/lfm-train/work/skills/tools.json --test $HOME/lfm-train/work/skills/test.jsonl --manifest $HOME/lfm-train/work/skills/manifest.json --model LiquidAI/LFM2.5-350M --model-revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f --label stock --launch --config $HOME/lfm-train/measure-config.toml --timeout 180 --out $HOME/lfm-train/work/measure/skills-stock.md`
- date: 2026-09-23
- endpoint: `<local endpoint>` (a local endpoint was used; never recorded)
- model: `LiquidAI/LFM2.5-350M`
- model revision: `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`
- device: <https://github.com/NVIDIA-AI-IOT/jetson-device-skills> at commit `20137897aef549cc2fa36a18c10e45da94967c3e`
- bsp: <https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills> at commit `fdfafef0416be1eb3852b68fc802a9752f33fb28`

## Results

| Split | Correct / total |
|---|---|
| overall | 35 of 104 (34%) |
| skill named in prompt | 23 of 34 (68%) |
| skill not named | 12 of 70 (17%) |
| repo: bsp | 24 of 48 (50%) |
| repo: device | 11 of 56 (20%) |

| Outcome | Count |
|---|---|
| correct | 35 |
| wrong_skill | 38 |
| no_call | 27 |
| several_calls | 4 |

| Latency | ms |
|---|---|
| median | 97 |
| p95 | 1066 |
