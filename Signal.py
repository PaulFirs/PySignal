from numpy import array, arange, exp, log10, abs as np_abs
from numpy.fft import rfft, rfftfreq, fft, fftfreq
from numpy.random import uniform
from math import cos, pi
import matplotlib.pyplot as plt
# а можно импортировать numpy и писать: numpy.fft.rfft
FD = 16000 # частота дискретизации, отсчётов в секунду
FD1 = 8000 # частота дискретизации, отсчётов в секунду
# а это значит, что в дискретном сигнале представлены частоты от нуля до 11025 Гц (это и есть теорема Котельникова)
N = 200 # длина входного массива, 0.091 секунд при такой частоте дискретизации
FSmin = 800.0
FSmax = 800.0
f0 = FSmin
b = (FSmax - FSmin)/(N/FD)/2
# сгенерируем сигнал с частотой 440 Гц длиной N
pure_sig  = array([10.*cos(2*pi*  (f0*(t/FD) + b*((t/FD)**2)) ) for t in range(N)])
pure_sig1 = array([3.*cos(2*pi*  (600*(t/FD) + b*((t/FD)**2)) ) for t in range(N)])
pure_sig2 = array([6.*cos(2*pi*  (440*(t/FD) + b*((t/FD)**2)) ) for t in range(N)])
# сгенерируем шум, тоже длиной N (это важно!)
noise = uniform(-1.,1., N)
pure_sig = pure_sig# + pure_sig1 + pure_sig2 + noise

alpha = 100
fading = array([(exp(-alpha*(t/FD))*10.*cos(2*pi*  (f0*(t/FD) + b*((t/FD)**2)) )) for t in range(N)])

# суммируем их и добавим постоянную составляющую 2 мВ (допустим, не очень хороший микрофон попался. Или звуковая карта или АЦП)
# sig = pure_sig# + 2.0 # в numpy так перегружена функция сложения
# вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,  constrained_layout=False)
# нарисуем всё это, используя matplotlib
# Сначала сигнал зашумлённый и тон отдельно
ax1.plot(arange(N)/float(FD), pure_sig) # по оси времени секунды!
# ax1.plot(arange(N)/float(FD1), pure_sig1, '--') # по оси времени секунды!
ax1.set_xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
ax1.set_ylabel(u'Напряжение, мВ')
ax1.set_title(u'Зашумлённый сигнал и тон 440 Гц')

# Сигнал затухающий
ax2.plot(arange(N)/float(FD), fading) # по оси времени секунды!
ax2.set_xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
ax2.set_ylabel(u'Напряжение, мВ')
ax2.set_title(u'Сигнал')

# Потом спектр
spectrum = rfft(pure_sig)
ax3.plot(rfftfreq(N, 1./FD), 20*log10(2*np_abs(spectrum)/N))
# rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
# нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
# делим на число элементов, чтобы амплитуды были в милливольтах, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
ax3.set_xlabel(u'Частота, Гц')
ax3.set_ylabel(u'Дб')
ax3.set_title(u'Спектр')

# Потом спектр
spectrum = rfft(fading)
ax4.plot(rfftfreq(N, 1./FD), 20*log10(2*np_abs(spectrum)/N))
# rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
# нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
# делим на число элементов, чтобы амплитуды были в милливольтах, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
ax4.set_xlabel(u'Частота, Гц')
ax4.set_ylabel(u'Дб')
ax4.set_title(u'Спектр затухающего')

plt.tight_layout()
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.show()