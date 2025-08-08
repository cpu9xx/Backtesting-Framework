from .Env import Env
from .events import EVENT
from .api import setting
from .logger import log
from .broker import UserTrade
from .object import TIME

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np

"""

风险指标的符号及计算公式参考：
https://www.joinquant.com/help/api/help#api:%E9%A3%8E%E9%99%A9%E6%8C%87%E6%A0%87

"""
class IndicatorPackage(object):
    def __init__(self):
        pass


class Recorder(object):
    def __init__(self):
        self._ucontext = None
        self._env = Env()
        self._returns_ls =[]
        self._Dp_ls = []
        self._Dm_ls = []
        self._bm_returns_ls = []
        self._hold_ls = []
        self._holding_days_ls = []
        self._dt_ls = []

        self._returns_dict = {}
        self._Dp_dict = {}
        self._hold_dict = {}
        self._holding_days_dict = {}

        self._bm_start_price = None

        self.trade_count_dict = {}
        self.gain_trades = {}
        self.loss_trades = {}

        self._transfer_ls = []
        self._transfer_dt_ls = []
        self._transfer_returnidx_ls = []
        # self._winning_count
        event_bus = self._env.event_bus
        event_bus.add_listener(EVENT.DAY_END, self._pnl)
        event_bus.add_listener(EVENT.FIRST_TICK, self._set_bm_start_price)
        event_bus.add_listener(EVENT.RECORD_TRADE, self._record_trade)
        event_bus.add_listener(EVENT.RECORD_TRANSFER, self._record_transfer)

    def set_user_context(self, ucontext):
        self._ucontext = ucontext

    def set_state(self, state):
        def decode_dict_keys(d):
            """递归还原 dict 中 key 是 '_int_xx' 的项"""
            decoded = {}
            for k, v in d.items():
                if isinstance(k, str) and k.startswith('_int_'):
                    k_new = int(k[5:])
                else:
                    k_new = k
                if isinstance(v, dict):
                    v = decode_dict_keys(v)
                decoded[k_new] = v
            return decoded
        
        import pandas as pd
        self._returns_ls = state['returns_ls']
        self._Dp_ls = state['Dp_ls']
        self._Dm_ls = state['Dm_ls']
        self._bm_returns_ls = state['bm_returns_ls']
        self._hold_ls = state['hold_ls']
        self._holding_days_ls = state['holding_days_ls']
        self._dt_ls = pd.to_datetime(state['dt_ls']).date.tolist()
        
        self._returns_dict = decode_dict_keys(state['returns_dict'])
        self._Dp_dict = decode_dict_keys(state['Dp_dict'])
        self._hold_dict = decode_dict_keys(state['hold_dict'])
        self._holding_days_dict = decode_dict_keys(state['holding_days_dict'])

        self._bm_start_price = state['bm_start_price']

        self.trade_count_dict = decode_dict_keys(state['trade_count_dict'])
        self.gain_trades = decode_dict_keys(state['gain_trades'])
        self.loss_trades = decode_dict_keys(state['loss_trades'])
        


    def get_state(self): 
        def encode_dict_keys(d):
            """递归地将字典中的 int 类型 key 加前缀"""
            encoded = {}
            for k, v in d.items():
                # 处理 key
                if isinstance(k, int):
                    k_str = f"_int_{k}"
                else:
                    k_str = k
                # 递归处理嵌套字典
                if isinstance(v, dict):
                    v = encode_dict_keys(v)
                encoded[k_str] = v
            return encoded
        return {
            'returns_ls': self._returns_ls,
            'Dp_ls': self._Dp_ls,
            'Dm_ls': self._Dm_ls,
            'bm_returns_ls': self._bm_returns_ls,
            'hold_ls': self._hold_ls,
            'holding_days_ls': self._holding_days_ls,
            'dt_ls': [date.strftime('%Y%m%d') for date in self._dt_ls],
            
            'returns_dict': encode_dict_keys(self._returns_dict),
            'Dp_dict': encode_dict_keys(self._Dp_dict),
            'hold_dict': encode_dict_keys(self._hold_dict),
            'holding_days_dict': encode_dict_keys(self._holding_days_dict),

            'bm_start_price': self._bm_start_price,

            'trade_count_dict': encode_dict_keys(self.trade_count_dict),
            'gain_trades': encode_dict_keys(self.gain_trades),
            'loss_trades': encode_dict_keys(self.loss_trades),
        }



    def get_index_price(self, security):
        dtime = self._env.current_dt
        if TIME.OPEN_AUCTION_END .value <= dtime.time() < TIME.BEFORE_CLOSE.value:
            field = 'open'
        else:
            field = 'close'
            if TIME.DAY_START.value <= dtime.time() < TIME.OPEN_AUCTION_END.value:
                dtime += datetime.timedelta(days=-1, hours=0, minutes=0)
        return self._env.index_data[security].loc[dtime.date():dtime.date(), [field]].iloc[-1][field]

    def _get_historical_returns(self, count, pindex):
        if pindex == -1:
            return self._returns_ls[-count:]
        else:
            return self._returns_dict[pindex][-count:]
        
    def _get_trade_count_info(self, security):
        if security:
            return self.trade_count_dict.get(security, [])
        else:
            return self.trade_count_dict 
    
    def _get_win_rate(self, pindex, spin):
        if spin < 5:
            raise ValueError("get_win_rate() spin should be greater than or equal to 5")
        recent_win_count = len(self.gain_trades.get(pindex, []))
        recent_loss_count = len(self.loss_trades.get(pindex, []))
        return recent_win_count/(recent_win_count+recent_loss_count) if (recent_win_count+recent_loss_count) >= 5 else 0.5
    
    def _get_pl_ratio(self, pindex, spin):
        if spin < 3:
            raise ValueError("get_pl_ratio() spin should be greater than or equal to 3")
        recent_win_trades = self.gain_trades.get(pindex, [])[-spin:]
        recent_loss_trades = self.loss_trades.get(pindex, [])[-spin:]
        if len(recent_win_trades) < 3 or len(recent_loss_trades) < 3:
            return 1.3, 1
        avg_win_profit = sum(recent_win_trades) / len(recent_win_trades)
        avg_loss_profit = sum(recent_loss_trades) / len(recent_loss_trades)
        return avg_win_profit, avg_loss_profit
    
    def _get_avg_hoding_days(self, pindex, spin):
        recent_hoding_days_ls = self._holding_days_dict.get(pindex, [])[-spin:]
        if recent_hoding_days_ls:
            return sum(recent_hoding_days_ls) / len(recent_hoding_days_ls)
        else:
            return 2

    def _get_expected_profit(self, pindex, spin):
        win_rate = self._get_win_rate(pindex, spin)
        avg_win_profit, avg_loss_profit = self._get_pl_ratio(pindex, spin)
        avg_holding_days = self._get_avg_hoding_days(pindex, spin)
        expected_profit = (win_rate * avg_win_profit - (1-win_rate) * avg_loss_profit) / avg_holding_days
        return expected_profit


    def _set_bm_start_price(self, event):
        if self._ucontext.current_dt and self._bm_start_price is None:
            self._bm_start_price = self.get_index_price(setting.get_benchmark())
        return False

    def _pnl(self, event):
        if self._bm_start_price:
            bm_price = self.get_index_price(setting.get_benchmark())
            bm_prev_return = self._bm_returns_ls[-1] if self._bm_returns_ls else 0
            bm_current_return = bm_price/self._bm_start_price - 1
            # current_return = self._ucontext.portfolio.returns
            # prev_return  = self._returns_ls[-1] if self._returns_ls else 0


            self._Dm_ls.append(bm_current_return - bm_prev_return)
            self._bm_returns_ls.append(bm_current_return)
            self._dt_ls.append(self._ucontext.current_dt.date())


            # total portfolio
            current_return = self._ucontext.subportfolios.returns
            prev_return  = self._returns_ls[-1] if self._returns_ls else 0
            self._Dp_ls.append(current_return - prev_return)
            self._returns_ls.append(current_return)
            self._hold_ls.append((self._ucontext.subportfolios.total_value - self._ucontext.subportfolios.available_cash) / self._ucontext.subportfolios.total_value)
            # self._hold_ls.append(self._ucontext.subportfolios.total_position_count)

            # each subportfolio
            for pindex, portfolio in enumerate(self._ucontext.subportfolios):
                current_return = portfolio.returns
                prev_return = self._returns_dict[pindex][-1] if self._returns_dict.get(pindex, []) else 0

                self._Dp_dict.setdefault(pindex, []).append(current_return - prev_return)
                self._returns_dict.setdefault(pindex, []).append(current_return)
                if portfolio.total_value == 0:
                    self._hold_dict.setdefault(pindex, []).append(0.0)
                else:
                    self._hold_dict.setdefault(pindex, []).append((portfolio.total_value - portfolio.available_cash) / portfolio.total_value)
        return False

    def _record_trade(self, event):
        trade = event.__dict__.get('trade')
        order = trade.order
        if order.record:
            dt = trade.time
            pindex = order.pindex
            holding_days = (dt - self._ucontext.subportfolios[pindex].positions[order.security].init_time).days
            self._holding_days_dict.setdefault(pindex, []).append(holding_days)
            self._holding_days_ls.append(holding_days)
            
            profit = 100*((order.price - order.avg_cost)/order.avg_cost)
            if profit > 0:
                self.gain_trades.setdefault(pindex, []).append(profit)
            else:
                self.loss_trades.setdefault(pindex, []).append(profit)
            self.trade_count_dict.setdefault(order.security, []).append(profit)
        return True
    
    def _record_transfer(self, event):
        transfer = event.__dict__.get('transfer')
        dt = transfer.time
        self._transfer_ls.append(transfer)
        self._transfer_dt_ls.append(dt.date())
        self._transfer_returnidx_ls.append(len(self._returns_ls))
        return True



    def fft(self, data_ls):
        data_array = np.array(data_ls)

        # 对时序数据进行快速傅里叶变换 (FFT)
        fft_result = np.fft.fft(data_array)

        # 计算频率分量对应的频率轴
        frequencies = np.fft.fftfreq(len(data_array))
        fft_result[np.abs(frequencies) > 50] = 0
        # 只保留正半轴 (非负频率部分)
        positive_frequencies = frequencies[:len(frequencies)//2]
        positive_fft_result = np.abs(fft_result[:len(fft_result)//2])

        # 忽略频率为 0 的部分，计算周期（1/频率）
        positive_frequencies = positive_frequencies[1:]  # 去掉频率为0的部分
        positive_fft_result = positive_fft_result[1:]  # 去掉对应的FFT结果

        # 计算周期
        periods = 1 / positive_frequencies

        return periods, positive_fft_result

    def is_delay(self, array1, array2, delay):
        if len(array1) != len(array2):
            return None
        corr_list = []
        correlation_matrix = np.corrcoef(array1, array2)
        correlation_coefficient = correlation_matrix[0, 1]
        # print(f"0: {correlation_coefficient}")
        corr_list.append(correlation_coefficient)
        for i in range(1, delay-1):
            correlation_coefficient = np.corrcoef(array1[:-i], array2[i:])[0, 1]
            corr_list.append(correlation_coefficient)

        return corr_list
    
    def autocorrelation(self, data, max_lag):
        data = np.asarray(data)
        mean = np.mean(data)
        n = len(data)
        corr_list = []
        for lag in range(1, max_lag-1):
            numerator = np.sum((data[:n-lag] - mean) * (data[lag:] - mean))
            denominator = np.sum((data - mean) ** 2)
            corr_list.append(numerator / denominator)
        
        return corr_list

    def cal_indicators(self, holding_days_ls, gain_trades:list, loss_trades:list, returns_ls, bm_returns_ls, hold_ls, dt_ls, Dp_ls, Dm_ls):
        if len(holding_days_ls) > 0:
            avg_holding_days = sum(holding_days_ls) / len(holding_days_ls)
        else:
            avg_holding_days = 0
        

        total_gain_count = len(gain_trades)
        total_gain_sum = sum(gain_trades)

        total_loss_count = len(loss_trades)
        total_loss_sum = sum(loss_trades)
        trade_count = total_gain_count + total_loss_count
        if trade_count > 0:
            winning_rate = total_gain_count / (trade_count)
        else:
            winning_rate = 0
        if total_loss_sum != 0:
            pl_ratio = abs(total_gain_sum / total_loss_sum)
        else:
            pl_ratio = np.inf

        avg_gain = total_gain_sum / total_gain_count if total_gain_count > 0 else 0
        avg_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else 0

        start_cash = self._env.usercfg['start_cash']
        total_return_rate = returns_ls[-1]
        bm_total_return_rate = bm_returns_ls[-1]

        Rp = ((1 + total_return_rate) ** (250/len(dt_ls)) - 1)
        Rm = ((1 + bm_total_return_rate) ** (250/len(dt_ls)) - 1)
        
        cov_matrix = np.cov(Dp_ls, Dm_ls)
        covDpDm = cov_matrix[0, 1]
        varDm = np.var(Dm_ls)

        hold_cov_matrix = np.cov(hold_ls, Dm_ls)
        covholdDm = hold_cov_matrix[0, 1]
        hold_beta = covholdDm / varDm

        Rf = 0.04
        beta = covDpDm / varDm
        alpha = Rp - (Rf + beta * (Rm - Rf))

        Op = np.sqrt(250 * np.var(Dp_ls, ddof=1))

        sharpe = (Rp - Rf) / Op
        sharpe2 = Rp / Op


        navs = np.array(returns_ls) + 1 
        peak = np.maximum.accumulate(navs)
        drawdown = (peak - navs) / peak     
        max_drawdown = np.max(drawdown)     
        max_drawdown_index = np.argmax(drawdown) 
        # peak = np.maximum.accumulate(returns_ls)
        # drawdown = peak - returns_ls
        # max_P_drawdown = np.max(drawdown) * start_cash
        # max_drawdown_index = np.argmax(drawdown)
        # # Py 最低点收益值
        # Py = (returns_ls[max_drawdown_index] + 1) * start_cash
        # Px = max_P_drawdown + Py
        # max_drawdown = (Px - Py)/Px

        
        if hold_ls:
            if isinstance(hold_ls[0], float):
                avg_hold = 100*np.mean(hold_ls)
            else:
                avg_hold = np.mean(hold_ls)
        else:
            avg_hold = 0

        indicator_pkg = IndicatorPackage()
        indicator_pkg.alpha = alpha
        indicator_pkg.beta = beta
        indicator_pkg.hold_beta = hold_beta
        indicator_pkg.avg_hold = avg_hold
        indicator_pkg.avg_holding_days = avg_holding_days
        indicator_pkg.winning_rate = winning_rate
        indicator_pkg.pl_ratio = pl_ratio
        indicator_pkg.avg_gain = avg_gain
        indicator_pkg.avg_loss = avg_loss
        indicator_pkg.sharpe = sharpe
        indicator_pkg.sharpe2 = sharpe2
        indicator_pkg.Op = Op
        indicator_pkg.trade_count = trade_count
        indicator_pkg.Rp = Rp
        indicator_pkg.Rm = Rm
        indicator_pkg.max_drawdown = max_drawdown
        indicator_pkg.max_drawdown_index = max_drawdown_index
        return indicator_pkg

    def plot(self, show=True, save_path='pnl.png'):
        if len(self._dt_ls) < 2:
            return
        fig = plt.figure(figsize=(8, 8))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
        ax = fig.add_subplot(gs[0])

        
        bm_return_array100 = 100*np.array(self._bm_returns_ls)
        returns_array100 = 100*np.array(self._returns_ls)

        if self._hold_ls and isinstance(self._hold_ls[0], float):
            max_total_return = max(returns_array100)
            max_sub_return = 100*max(max(v) for v in self._returns_dict.values() if v) 
            max_return = max(max_total_return, max_sub_return)
            max_bm_return = max(bm_return_array100)
            scale_coff = max(10, max(max_bm_return, max_return))
        else:
            scale_coff = 1
        hold_array_scale = scale_coff*np.array(self._hold_ls)
        line3,  = ax.plot(self._dt_ls, hold_array_scale, label="Total Pos Ratio", color="#ECB051", linewidth=2, picker=5, zorder=9)
        line2,  = ax.plot(self._dt_ls, bm_return_array100, label="Benchmark Returns", color="red", linewidth=2, picker=5, zorder=10)
        line1,  = ax.plot(self._dt_ls, returns_array100, label="Returns", color="blue", linewidth=2, picker=5, zorder=10)
        sub_return_line_dict = {}
        for pindex in self._returns_dict.keys():
            # color_dark, color_light = color_pairs[key % len(color_pairs)]
            returns_ls = self._returns_dict[pindex]
            holds_array = scale_coff*np.array(self._hold_dict[pindex])
            sub_return_line,  = ax.plot(self._dt_ls, 100*np.array(returns_ls), label=f"P{pindex} Returns", linewidth=1.5, picker=5, zorder=2)
            sub_return_line_dict[pindex] = sub_return_line
            # ax.plot(self._dt_ls, holds_array, label=f"P{pindex} Pos Ratio", color=sub_return_line.get_color(), linewidth=1, picker=5)
            # ax.fill_between(self._dt_ls, holds_array, color=color_light, alpha=0.9)


        ax.fill_between(self._dt_ls, returns_array100, color="#B9CFE9", alpha=0.5, zorder=1)
        ax.fill_between(self._dt_ls, hold_array_scale, color="#ECB762", alpha=0.5, zorder=1)


        transfer_points, = ax.plot(self._transfer_dt_ls, [returns_array100[idx] for idx in self._transfer_returnidx_ls], 'x', color='black', picker=5, markeredgewidth=1.5, zorder=10)
        global enlarge_transfer
        enlarge_transfer = None

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # ax.set_title("PNL", fontsize=16)
        # ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Returns (%)", fontsize=12)

        # 自动格式化 x 轴标签
        fig.autofmt_xdate()
        ax.legend(loc='upper left', fontsize=8, frameon=False, framealpha=0.8)

        annot = ax.annotate(
            "", 
            xy=(0, 0), 
            xytext=(20, 20), 
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=0.5),
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.1", color="gray"),
            zorder=100
        )

        annot.set_visible(False)

        text_dict = {}
        indicator_pkg_dict = {}
        def get_text(pkg):
            text_str = f"Alpha: {pkg.alpha:.2f}   Beta(hold): {pkg.beta:.2f}({pkg.hold_beta:.2f})   Avg hold(days): {pkg.avg_hold:.2f} %({pkg.avg_holding_days:.1f})   Win rate: {100*pkg.winning_rate:.2f}%   PL ratio: {pkg.avg_gain:.2f} : {abs(pkg.avg_loss):.2f}"
            text_str2 = f"\nSharpe: {pkg.sharpe:.2f}({pkg.sharpe2:.2f})     Volatility: {pkg.Op:.2f}    Trade count: {pkg.trade_count}    Rp: {100*pkg.Rp:.2f}%    Rm: {100*pkg.Rm:.2f}%    Max drawdown: {100*pkg.max_drawdown:.2f}%"            
            return text_str + text_str2
        
        for pindex in self._returns_dict.keys():
            pkg = self.cal_indicators(holding_days_ls=self._holding_days_dict.get(pindex, []),
                                        gain_trades=self.gain_trades.get(pindex, []), 
                                        loss_trades=self.loss_trades.get(pindex, []), 
                                        returns_ls=self._returns_dict[pindex], 
                                        bm_returns_ls=self._bm_returns_ls, 
                                        hold_ls=self._hold_dict[pindex], 
                                        dt_ls=self._dt_ls, 
                                        Dp_ls=self._Dp_dict[pindex], 
                                        Dm_ls=self._Dm_ls)
            indicator_pkg_dict[pindex] = pkg
            text_dict[pindex] = get_text(pkg)

        main_pkg = self.cal_indicators(holding_days_ls=self._holding_days_ls,
                                        gain_trades=[item for sublist in self.gain_trades.values() for item in sublist], 
                                        loss_trades=[item for sublist in self.loss_trades.values() for item in sublist], 
                                        returns_ls=self._returns_ls, 
                                        bm_returns_ls=self._bm_returns_ls, 
                                        hold_ls=self._hold_ls, 
                                        dt_ls=self._dt_ls, 
                                        Dp_ls=self._Dp_ls, 
                                        Dm_ls=self._Dm_ls)
        text_dict['main'] = get_text(main_pkg)
        ax.scatter(self._dt_ls[main_pkg.max_drawdown_index], 100*self._returns_ls[main_pkg.max_drawdown_index], color='black', s=25, zorder=15)
        title = ax.text(
            -0.10, 1.2,
            text_dict['main'],
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='white')
        )


        def update_title(pindex):
            title.set_text(text_dict[pindex])

        def update_annot(obj, ind, type):
            if type == 'pnl':
                pos = obj.get_xydata()[ind["ind"][0]]
                annot.xy = pos
                date_str = mdates.num2date(pos[0]).strftime("%Y-%m-%d")
                padding_length = max(0, len(f"Date:  {date_str}") - len(f"Returns:  {pos[1]:.2f}%"))
                padding = ' ' * (padding_length+1)
                text = f"Date:  {date_str}\nReturns:  {padding}{pos[1]:.2f} %"
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.9)  # 设置背景透明度，使其略显高亮
            elif type == 'transfer':
                global enlarge_transfer
                idx = ind["ind"][0]
                transfer = self._transfer_ls[idx]
                pos = obj.get_xydata()[idx]
                annot.xy = pos
                text = f"Transfer date:        {transfer.time.strftime('%Y-%m-%d')}\nP{transfer.info[1]}: {transfer.info[3]:,}  - {transfer.info[0]:,}\nP{transfer.info[2]}: {transfer.info[4]:,} + {transfer.info[0]:,}"
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.9)
                # print(enlarge_transfer)
                if enlarge_transfer is not None:
                    enlarge_transfer.set_visible(True)
                else:
                    enlarge_transfer = ax.plot(pos[0], pos[1], 'x', color='black', markersize=15, markeredgewidth=2, zorder=11)[0]

        def hover(event):
            # 如果鼠标在坐标轴范围内
            if event.inaxes == ax:
                # 记录特殊事件的散点的注释会优先于 pnl 显示
                cont_transfer, ind = transfer_points.contains(event)
                cont1, ind1 = line1.contains(event)
                cont2, ind2 = line2.contains(event)

                if cont_transfer:
                    update_annot(transfer_points, ind, type='transfer')
                    annot.set_visible(True)
                elif cont1:
                    update_annot(line1, ind1, type='pnl')
                    annot.set_visible(True)
                elif cont2:
                    update_annot(line2, ind2, type='pnl')
                    annot.set_visible(True)
                else:
                    for pindex, line in sub_return_line_dict.items():
                        cont, ind = line.contains(event)
                        if cont:
                            update_title(pindex)
                            line.set_zorder(11)
                            # title.set_visible(True)
                            break
                        else:
                            line.set_zorder(2)
                    else:
                        update_title('main')
                        annot.set_visible(False)
                        global enlarge_transfer
                        if enlarge_transfer is not None:
                            enlarge_transfer.set_visible(False)
                        
                fig.canvas.draw_idle()

        # 连接鼠标移动事件和 hover 函数
        fig.canvas.mpl_connect("motion_notify_event", hover)

            
        # text_str = f"Alpha: {alpha:.2f}    Beta(hold): {beta:.2f}({hold_beta:.2f})    Avg hold(days): {avg_hold:.2f} %({avg_holding_days:.1f})     Win rate: {100*winning_rate:.2f}%    PL ratio: {pl_ratio:.2f} : 1"
        # text_str2 = f"\nSharpe: {sharpe:.2f}    Volatility: {Op:.2f}    Trade count: {trade_count}    Rp: {100*Rp:.2f}%    Rm: {100*Rm:.2f}%    Max drawdown: {100*max_drawdown:.2f}%"

        
        # ax.text(-0.10, 1.2, text_str+text_str2, transform=ax.transAxes, fontsize=10,
        #     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='white'))
        
        # corr_ls = self.is_delay(self._hold_ls, self._hold_ls, len(self._hold_ls))
        corr_ls = self.autocorrelation(self._hold_ls, len(self._hold_ls))

        # periods, positive_fft_result = self.fft(self._hold_ls)
        
        fftax = fig.add_subplot(gs[1])
        line_fft = fftax.plot(corr_ls, label="Corrcoef delay", color="blue", linewidth=1)
        # line_fft = fftax.plot(periods, positive_fft_result, label="Hold count FFT", color="blue", linewidth=1)
        fftax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fftax.set_title("Position self corrcoef", fontsize=10)

        plt.subplots_adjust(hspace=0.8) 
        plt.savefig(save_path)
        if show:
            plt.show()

            import csv

            with open('trade_stats.csv', 'w', newline='') as f:
                writer = csv.writer(f)

                # Header with percentile columns
                header = ['Security', 'Count', 'Total Profit', 'Average Profit'] + [f'P{p}' for p in range(0, 101, 10)]
                writer.writerow(header)

                for security, profits in self.trade_count_dict.items():
                    if not profits:
                        row = [security, 0, 0, 0] + [0] * 11
                    else:
                        total = sum(profits)
                        avg = total / len(profits)
                        percentiles = np.percentile(profits, list(range(0, 101, 10)))
                        row = [security, len(profits), total, avg] + list(percentiles)
                    writer.writerow(row)
