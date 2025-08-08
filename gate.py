import pandas as pd
import os
from sqlalchemy import create_engine, text, types
from sqlalchemy.orm import sessionmaker
import multiprocessing


class MySQL_gate():
    def __init__(self, start_date=None, end_date=None, **kwargs):
        self.start_date = start_date
        self.end_date = end_date
        self.db_config = kwargs
        self.session_dict = {}
        self.StockSession = None
        self.IndexSession = None
        self.LimitSession = None
        self.HmDetailSession = None
        self.ImpDividendSession = None

    def init_session(self, db):
        engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{db}", pool_size=10)
        self.session_dict[db] = sessionmaker(bind=engine)()

        # stock_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/stock", pool_size=5)
        # index_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/index", pool_size=5)
        # limit_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/limit", pool_size=5)
        # hm_detail_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/hm_detail?charset=utf8mb4", pool_size=5)
        # cb_daily_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/cb_daily", pool_size=5)
        # imp_dividend_engine = create_engine(f"mysql+mysqldb://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/imp_dividend", pool_size=5)

        # self.StockSession = sessionmaker(bind=stock_engine)
        # self.IndexSession = sessionmaker(bind=index_engine)
        # self.LimitSession = sessionmaker(bind=limit_engine)
        # self.HmDetailSession = sessionmaker(bind=hm_detail_engine)
        # self.CBDailySession = sessionmaker(bind=cb_daily_engine)
        # self.ImpDividendSession = sessionmaker(bind=imp_dividend_engine)
        
        current_process = multiprocessing.current_process()
        print(f"{current_process.name} init_session db: {db}")
        return self.get_session(db)
        
    def get_session(self, db):
        try:
            if db in self.session_dict:
                return self.session_dict[db]
            else:
                return self.init_session(db)
            # if db == 'stock':
            #     if self.StockSession is not None:
            #         return self.StockSession()
            #     else:
            #         return self.init_session(db)
            # elif db == 'index':
            #     if self.IndexSession is not None:
            #         return self.IndexSession()
            #     else:
            #         return self.init_session(db)
            # elif db == 'limit':
            #     if self.LimitSession is not None:
            #         return self.LimitSession()
            #     else:
            #         return self.init_session(db)
            # elif db == 'hm_detail':
            #     if self.HmDetailSession is not None:
            #         return self.HmDetailSession()
            #     else:
            #         return self.init_session(db)
            # elif db == 'cb_daily':
            #     if self.CBDailySession is not None:
            #         return self.CBDailySession()
            #     else:
            #         return self.init_session(db)
            # elif db == 'imp_dividend':
            #     if self.ImpDividendSession is not None:
            #         return self.ImpDividendSession()
            #     else:
            #         return self.init_session(db)
            # else:
            #     print(f"Error get_session with (db: {db})")
            #     return None
        except Exception as e:
            print(f"Error get_session ('{db}'): {e}")
            return self.init_session(db)

    def get_tables(self, db='stock'):
        query = f"SHOW TABLES;"
        df = pd.DataFrame(self.execute_query(query, db, commit=False))
        df = df.sort_values(df.columns[0])
        return df

    def read_data_date(self, tablename, db='stock', days=1):
        query = f"SELECT * FROM `{tablename}` ORDER BY trade_date DESC LIMIT {days}"
        df = pd.DataFrame(self.execute_query(query, db, commit=False))
        return df
    
    def read_df(self, tablename, db='stock'):
        query = f"SELECT * FROM `{tablename}` WHERE trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}';"
        df = pd.DataFrame(self.execute_query(query, db, commit=False))
        return df
    
    def read_entire_df(self, tablename, db='limit'):
        query = f"SELECT * FROM `{tablename}`;"
        df = pd.DataFrame(self.execute_query(query, db, commit=False))
        return df
    
    def read(self, tablename, db='stock'):
        query = f"SELECT * FROM `{tablename}` WHERE trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}';"
        return self.execute_query(query, db, commit=False)

    def execute_query(self, query, db, commit=False):
        session = self.get_session(db)

        try:
            result = session.execute(text(query))
            if commit:
                session.commit()
            if result.returns_rows:
                res = result.fetchall()
                return res
            else:
                return result.rowcount
        except Exception as e:
            session.rollback()
            print(f"Error executing query('{query}'): {e}")
            return None
        finally:
            session.close()
        
