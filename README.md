
## Strategy
Develop your strategy in userStrategy.py. If it is the first time loading data, set the "if_load_data" in "userconfig" dict to False. Once you load, new pickle files will be created, it is used for save data loading time. Then you can directly load data from 'data.pkl' rather than the database by setting "if_load_data" to True. Run the following to backtest:
```
python run.py
```

# Requirements 
- python>=3.8.10
- matplotlib==3.7.1
- pandas==1.4.2
- numpy==1.24.4
- SQLAlchemy==2.0.35

