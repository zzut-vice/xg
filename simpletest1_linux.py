'''
start: 2021-01-01 00:00:00
end: 2021-05-20 00:00:00
period: 1h
exchanges: [{"eid":"Binance","currency":"BTC_USDT","stocks":0}]
'''
from fmz import * # 导入所有FMZ函数
import pandas as pd
task = VCtx(__doc__) # 初始化
save_name = 'rowdata_test1.csv'

exchange.GetAccount()
kline = get_bars('bitfinex.btc_usd', '1m', start='2019-05-01 06:00:00', end='2019-05-01 20:00:00')

start_time = _D()
Sleep(240*60*1000)
end_time = _D()
kline_all = get_bars('bitfinex.btc_usd', '1m', start=start_time, end=end_time)
kline_all['EMAfast'] = talib.EMA(kline.close,5)
kline_all['EMAslow'] = talib.EMA(kline.close,12)
while kline_all.shape[0]<1000:
    start_time = end_time
    Sleep(60*60*1000)
    end_time = _D()
    kline = get_bars('bitfinex.btc_usd', '1m', start=start_time, end=end_time)
    kline_all = pd.concat([kline_all,kline],axis=0)
slow_length = 12
kline_all['EMAfast'] = talib.EMA(kline_all.close,5)
kline_all['EMAslow'] = talib.EMA(kline_all.close,slow_length)
kline_all[slow_length-1:].to_csv(save_name) # to_csv 每1000行 占用 98kB

print(end_time)
