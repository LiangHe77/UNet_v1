from datetime import datetime
import time
a = '20120405120304'
b = '20120405120902'
time1 = datetime.strptime(a, '%Y%m%d%H%M%S')
time2 = datetime.strptime(b, '%Y%m%d%H%M%S')
delta = (time2-time1).seconds
print(delta/60.0)
