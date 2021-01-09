import numpy as np
from pandas import date_range
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, drange 
from pylab import legend
from datetime import datetime, timedelta
import sys

from euler import iterative_method, dSIR, dSIRD

def plot_ir_data(i, r, t=None, t0=datetime(2020, 1, 22), dt=timedelta(days=1)):
    if t is None:
        t = drange(t0, t0 + i.shape[0] * dt, dt)
    
    fig, ax = plt.subplots() 
    ax.plot_date(t, i, 'r-o', label="Nhiễm bệnh") 
    ax.plot_date(t, r, 'b-x', label="Hồi phục")
    
    fig.autofmt_xdate()
    
    ax.set_title("Giá trị I, R thực")
    legend(loc='upper left')
    
    plt.show()


def plot_ird_data(i, r, d, t=None, t0=datetime(2020, 1, 22), dt=timedelta(days=1)):
    if t is None:
        t = drange(t0, t0 + i.shape[0] * dt, dt)
    
    fig, ax = plt.subplots() 
    ax.plot_date(t, i, 'r-o', label="Nhiễm bệnh") 
    ax.plot_date(t, r, 'b-x', label="Hồi phục")
    ax.plot_date(t, d, 'g-*', label="Tử vong")
    
    fig.autofmt_xdate()
    
    ax.set_title("Giá trị I, R thực")
    legend(loc='upper left')
    
    plt.show()


def plot_sir_prediction(s0=796702206,
                i0=2,
                r0=0,
                beta=0.1126,
                gamma=0.0252,
                t0=datetime(2020, 1, 22),
                dt=timedelta(days=1),
                nStep=100,
                t=None,
                method='Euler'):
    
    method = method.title()
    
    dsir = lambda x0, t0, dt : dSIR(s=x0[0], i=x0[1], r=x0[2], dt=dt, beta=beta, gamma=gamma)
    
    if method == 'Improved Euler':
        method = lambda s0, i0, r0, t: iterative_method((s0, i0, r0), dsir, t, method='Improved Euler')
        methodName = "Euler mở rộng"
    elif method == 'Rk4':
        method = lambda s0, i0, r0, t: iterative_method((s0, i0, r0), dsir, t, method='Rk4')
        methodName = "Runge-Kutta bậc 4"
    else:
        method = lambda s0, i0, r0, t: iterative_method((s0, i0, r0), dsir, t, method='Euler')
        methodName = "Euler"
        
    if t is None:
        t = drange(t0, t0 + (nStep + 1) * dt, dt)
    
    sir = method(s0, i0, r0, t)
    print(sir)
    i = np.rint(sir[:, 1]).astype(int)
    r = np.rint(sir[:, 2]).astype(int)
    
    print("  {:} {: >15} {: >20} ".format("Ngày","Ca nhiễm","Ca hồi phục"))
    dates = date_range(t0, t0 + nStep * dt, nStep + 1)
    for date, i_t, r_t in zip(dates, i, r):
        print("{:>5} {: >20} {: >20} ".format(date.strftime("%d/%m"), i_t, r_t))
    
    fig, ax = plt.subplots()
    ax.plot_date(t, i, 'r-o', label="Nhiễm bệnh")
    ax.plot_date(t, r, 'b-x', label="Hồi phục")
    
    fig.autofmt_xdate()
    
    ax.set_title("Sử dụng giải thuật " + methodName + " để xấp xỉ các đại lượng I, R")
    legend(loc='upper left')
    
    plt.show()
    

if __name__ == "__main__":
    if len(sys.argv) < 7:
        plot_sir_prediction(method='RK4')
    elif len(sys.argv) == 7:
        plot_sir_prediction(s0=sys.argv[1], i0=sys.argv[2], r0=sys.argv[3], \
                    beta=sys.argv[4], gamma=sys.argv[5], nStep=sys.argv[6])
    else:
        plot_sir_prediction(s0=sys.argv[1], i0=sys.argv[2], r0=sys.argv[3], \
            beta=sys.argv[4], gamma=sys.argv[5], nStep=sys.argv[6], method=sys.argv[7])