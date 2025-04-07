# imc_prosperity3

## Setup
```
pip install -U prosperity3bt
```


## Usage
```
# Run the backtester on an algorithm using all data from round 1
prosperity3bt example_program.py 1
```

Then it will output something like:
```
Backtesting /Users/yuitora./prospe/example_program.py on round 1 day -2
100%|###############################################################################################################################################################################| 10000/10000 [00:00<00:00, 14233.46it/s]
KELP: 0
RAINFOREST_RESIN: 8,192
SQUID_INK: 0
Total profit: 8,192

Backtesting /Users/yuitora./prospe/example_program.py on round 1 day -1
100%|###############################################################################################################################################################################| 10000/10000 [00:00<00:00, 15000.49it/s]
KELP: 0
RAINFOREST_RESIN: 7,428
SQUID_INK: 0
Total profit: 7,428

Backtesting /Users/yuitora./prospe/example_program.py on round 1 day 0
100%|###############################################################################################################################################################################| 10000/10000 [00:00<00:00, 14876.97it/s]
KELP: 0
RAINFOREST_RESIN: 8,132
SQUID_INK: 0
Total profit: 8,132

Profit summary:
Round 1 day -2: 8,192
Round 1 day -1: 7,428
Round 1 day 0: 8,132
Total profit: 23,752

Successfully saved backtest results to backtests/2025-04-07_21-17-51.log
```

On the last line, you can see that it saves the result to a `.log` file in the folder `backtests/`.

You can upload that file to this website to visualize the result: https://jmerle.github.io/imc-prosperity-3-visualizer/.

