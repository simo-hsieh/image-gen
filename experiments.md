# Experiment Log

## Best Settings So Far
- **Checkpoint**: ghibli_diffusion.ckpt
- **Denoise**: 0.6
- **ControlNet**: 0.7
- **Steps**: 25
- **CFG**: 9
- **Sampler**: dpmpp_2m_sde / karras
- **Max size**: 512
- **Best seeds**: 42, 456

## Experiments

### v1 — Baseline (batch_v1)
- **Changed**: First run with toonyou_beta6, default prompts
- **Settings**: denoise=0.45, cn=0.85, steps=15, cfg=7, 512x512 square
- **Result**: Very subtle effect, barely stylized
- **Learned**: toonyou checkpoint too weak for ghibli, denoise too low

### v2 — Structure improvements (batch_v2)
- **Changed**: Dedicated ghibli_diffusion.ckpt, lineart preprocessor, aspect ratio fix, dpmpp_2m_sde sampler
- **Settings**: denoise=0.5/0.65/0.8, cn=0.8, steps=15, cfg=7
- **Result**: Better structure, correct aspect ratio, but face not handsome
- **Learned**: Lineart preprocessor + aspect ratio = big quality jump

### v3 — Handsome prompt (batch_v3)
- **Changed**: Added "handsome, sharp jawline, clear skin, bright eyes" to prompt
- **Settings**: denoise=0.5/0.65, cn=0.8, steps=25, cfg=9
- **Result**: Face looks better, more attractive
- **Learned**: Quality tags in prompt matter a lot

### v4 — Protagonist prompt (batch_v4)
- **Changed**: Added "protagonist, ikemen, bishounen, soft facial features, gentle face"
- **Settings**: denoise=0.55/0.65, cn=0.8, steps=25, cfg=9
- **Result**: Best face so far — soft, main character vibe
- **Learned**: "ikemen, bishounen" + "soft facial features" = anime protagonist look

### v5 — Clothing prompt (batch_v5) FAILED
- **Changed**: Added clothing descriptors "neat simple dark jacket, clean clothing, tidy outfit"
- **Settings**: denoise=0.55/0.6/0.65, cn=0.7/0.8, steps=25, cfg=9
- **Result**: Way worse than v4 — face quality dropped, style weakened
- **Learned**: Too many clothing keywords dilute face quality. Don't over-prompt.

### v6 — v4 prompt + param tuning (batch_v6)
- **Changed**: Reverted to v4 prompt, tuned denoise/CN instead
- **Settings**: denoise=0.55/0.6, cn=0.7/0.8, steps=25, cfg=9
- **Result**: d060_cn07_s42 best — cleaner coat naturally, good face
- **Learned**: Lower CN (0.7) lets model clean up coat without prompt spam

### v7 — Black jacket (batch_v7)
- **Changed**: Added single term "black jacket" to prompt
- **Settings**: denoise=0.6, cn=0.7, steps=25, cfg=9
- **Result**: Darker coat, face quality preserved
- **Learned**: Single clothing term works, unlike v5's multi-term approach

### v8 — Detail tuning (batch_v8_*)
- **Changed**: 4 parallel experiments
  - `batch_v8_higher_res`: 768px (sharper details?)
  - `batch_v8_more_steps`: 35 steps (smoother rendering?)
  - `batch_v8_cfg_tuning`: cfg=8 (softer/natural?)
  - `batch_v8_more_seeds`: 5 new seeds (better variations?)
- **Settings**: base = denoise=0.6, cn=0.7, steps=25, cfg=9
- **Result**: Pending review
- **Learned**: TBD
