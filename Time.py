import numpy as np
import datetime
import math



def classify_onehot(t_str):
    t_dt = datetime.datetime.strptime(t_str,'%Y-%m-%d %H:%M:%S')
    t_dt_weekday = t_dt.weekday()
    t_dt_hour = t_dt.hour
    t_dt_h = math.floor(t_dt_hour/3)
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==0:
        return [1,0,1,0,0,0,0,0,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==1:
        return [1,0,0,1,0,0,0,0,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==2:
        return [1,0,0,0,1,0,0,0,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==3:
        return [1,0,0,0,0,1,0,0,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==4:
        return [1,0,0,0,0,0,1,0,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==5:
        return [1,0,0,0,0,0,0,1,0,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==6:
        return [1,0,0,0,0,0,0,0,1,0]
    if t_dt_weekday>=0 and t_dt_weekday<=4 and t_dt_h ==7:
        return [1,0,0,0,0,0,0,0,0,1]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==0:
        return [0,1,1,0,0,0,0,0,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==1:
        return [0,1,0,1,0,0,0,0,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==2:
        return [0,1,0,0,1,0,0,0,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==3:
        return [0,1,0,0,0,1,0,0,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==4:
        return [0,1,0,0,0,0,1,0,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==5:
        return [0,1,0,0,0,0,0,1,0,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==6:
        return [0,1,0,0,0,0,0,0,1,0]
    if t_dt_weekday == 5 or t_dt_weekday == 6 and t_dt_h ==7:
        return [0,1,0,0,0,0,0,0,0,1]
