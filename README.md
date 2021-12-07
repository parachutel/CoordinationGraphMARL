## Research Doc

https://docs.google.com/document/d/1inCNIR2s-0SK7oPD-l3PR0ItI0t4kIBA8F43G7V9BMs/edit?usp=sharing

## Run Training
Launch **basic** training with command: 
`python cg_runner.py --policy <policy> --env <env>`.
Progress logging and checkpoints can be found in the `/save` directory.

 For other argument options, look into `args.py`, `env_maker.py` and `policy_maker.py`.

Currently available `<policy>`: 
- `de` (decentralized)
- `dicg_ce` (DICG-CE)
- `proximal_cg` (proximity-based coordination graph)

Currently available `<env>`: 
- `meet` (meeting in the grid world)
- `predprey` (predator-prey)
- `traffic` (hard mode traffic junction)