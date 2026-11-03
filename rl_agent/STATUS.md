# RL experiment complete — verification passed

Six offline fitted-Q agents trained: return, loss-averse return, and precision rewards, each with two random seeds. Training used 4,399 complete-endpoint trajectories from the first 300 sessions. All choices were made on earlier validation periods before RL test scoring.

Selected policy: return_seed731, entry margin 0.0. It failed the 80% precision-validation target and remains a research-only fallback.

Later-period evaluation (2026-07-15 through 2026-10-28): 1,018 scorable stock-days; 322 closed trades (156 wins, 166 non-winners), 613 confirmed no-trades, 83 unresolved episodes. Known win precision 48.4%; conservative precision 38.5%. Mean known trade net -0.05% after an assumed 0.5% round-trip cost. On the identical complete-endpoint subset it did not beat remaining in cash.

Loss-averse and precision-focused agents made no trades on complete test trajectories. The model can wait, enter, hold and exit; later actions have next-landmark delayed fills. Decisions are coarse horizon steps, not minute-level. The prior studies already viewed these test dates, so this is an exploratory evaluation, not independent future confirmation.

Report and graph: results/REPORT.md, results/rl_overview.png.
Saved models: results/agents/.
Readable selected trades and decisions: results/selected_trades.csv, results/selected_actions.csv.

Five environment/reward tests passed. Source hashes, split membership, valid actions, delayed fills, trade accounting, unchanged observations under future-price perturbations, learned-Q action stability, and saved-agent replay all verified. Original source data remains unchanged.

See README.md for reproduction and limitations. Do not select or tune another agent on the already viewed test results and call it a fresh test.
