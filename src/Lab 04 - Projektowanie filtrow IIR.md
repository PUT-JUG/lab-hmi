<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Transformata Fouriera sygnałów dyskretnych- splot, modulacja, okna

## Wstęp

### Filtry IIR (Infinite Impulse Response)
Jest to klasa filtrów rekursywnych, czyli posiadających sprzężeń zwrotnych. Oznacza to że odpowiedź filtru wyznaczana jest jako kombinacja liniowa pewnej ilości próbek sygnału wejściowego i wyjściowego. Transmitancja ma postać:
$$
H(z)=\frac{a_{0}+a_{1} z^{-1}+\ldots+a_{p} z^{-p}}{1+\left(b_{1} z^{-1}+\ldots+b_{q} z^{-q}\right)}
$$
a zapis równania różnicowego:
$$
y(n) = a_{0} x(n)+a_{1} x(n-1)+\ldots+a_{p} x(n-p) - b_{1} y(n-1)+\ldots+b_{p} y(n-p)
$$
gdzie \\(x(n)\\) jest sygnałem wejściowym, \\(y(n)\\) sygnałem wyjściowym.

### Typy filtrów
Zazwyczaj do projektowania filtrów IIR używa się tzw. prototypów określających strukturę filtra determinującą jego parametry w paśmie przepustowym i zaporowym, a następnie prototyp dostosowuje się do wymagań projektowych (tłumienia, zafalowań w paśmie przepustowym, stomości pasma przejściowego, typu filtra (DP, Gp, PP, PZ)
Najczęściej używanymi prototypami są prototyp Butterwortha, Czebyszewa I i II rodzaju oraz filtr eliptyczny, Bessela. Prototypy te mają różne właściwości, więcej na ich temat możesz znaleźć w materiałach z wykładu. Poniżej zebrano tylko najważniejsze cechy:

|              | paśmo przepustowe   | pasmo zaporowe      | stromość                | nieliniowość charakterystyki            |
|--------------|---------------------|---------------------|-------------------------|-----------------------------------------|
| Butterowth   | ch. monotoniczna    | ch. monotoniczna    | N*6dB/oct               | najbardziej liniowa spośród filtrów IIR |
| Chebyszew I  | ch. niemonotoniczna | ch. monotoniczna    | lepsza niż Butterwortha | silniejsza niż Czebyszewa II            |
| Czebyszew II | ch. monotoniczna    | ch. niemonotoniczna | lepsza niż Butterwortha | silniejsza niż Butterwortha             |
| Eliptyczny   | ch. niemonotoniczna | ch. niemonotoniczna | lepsza niż Czebyszewa   | Dużo silniejsza niż Czebyszewa II       |

### Metody projektowania filtrów
Poniżej przedstawiono procedurę projektowania filtrów:
1. Przeanalizowanie wymagań częstotliwościowych stawianych filtrowi i wybór odpowiedniego rodzaju filtra prototypowego i wybór odpowiedniego filtra prototypowego (Butterowtha, Czebyszewa typu I, czebyszewa typu II, eliptycznego)
2. Przeliczenie wymagań projektowych, stawianych filtrowi na rząd filtra
3. Zaprojektowanie transmitancji filtra
4. Sprawdzenie właściwości zaprojektowanego układu.

W Pythonie do oszacowaniu rzaeu filtrai zaprojektowania transmitancji  można wykorzystać następujące funkcje:
|              | parametry projektowe       | wyznaczenie rzędu                              | wyznaczenie transmitancji                             |
|--------------|----------------------------|------------------------------------------------|-------------------------------------------------------|
| Butterowth   | N, fn                      | N, fn = signal.buttord(fp, fz, Rp, Rs, fs=fs)  | b, a = signal.butter(N, fn, btype='low', fs=fs)       |
| Chebyszew I  | N, Rp, fn                  | N, fn = signal.cheb1ord(fp, fz, Rp, Rs, fs=fs) | b, a = signal.cheby1(N, Rp, fn, btype='low', fs=fs)   |
| Czebyszew II | N, Rp, fn                  | N, fn = signal.cheb2ord(fp, fz, Rp, Rs, fs=fs) | b, a = signal.cheby2(N, Rp, fn, btype='low', fs=fs)   |
| Eliptyczny   | N, Rp,Rs, fn - rząd filtra | N,fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)  | b, a = signal.ellip(N, Rp,Rs, fn, btype='low', fs=fs) | 


Parametry projektowe:
 - fp - częstotliwość graniczna pasam przepustowego
 - fz - częstotliwość graniczna pasam zaporowego
 - Rp - minimalne wzmocnienie w paśmie zaporowym w [dB] - wyrażone jako 20log(1-max(rp)) 
 - Rs - tłumienie w paśmie przepustowym i zaporowym
 - fs - częstotliwość próbkowania
 - N - rząd filtra
 - btype={‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
 - fn - czestotliwość tłumienia -3dB


``` python

import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt


fc = 3000
fz = 4500
rp = 1
rs = 60
N = 8
def plot_response(w, hf, hfb=None, label=''):
    fig, ax = plt.subplots(tight_layout=True)
    ax.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    if hfb is not None:
        ax.plot(w, hfb, c='#c0c0c0', label='Butterworth')
    ax.plot(w, hf, label=label)
    ax.grid()
    ax.legend()
    ax.set_xlim(0, 12000)
    ax.set_ylim(-120, 5)
    ax.set_xlabel('Częstotliwość [Hz]')
    ax.set_ylabel('Wzmocnienie [dB]')
    # ax.set_title('Charakterystyka częstotliwościowa filtru IIR')
    return ax

# filtr Butterwortha
fp = 3000 # fpass [Hz]
fz = 4500 # fstop [Hz]
Rp = 1 # -1dB w paśmie przepustowym
Rs = 60 # -60db w paśmie zaporowym
fs = 48000 # częstotliwość próbkowania


N, fn = signal.buttord(fp, fz, Rp, Rs, fs=fs)
bb, ab = signal.butter(N, fn, 'low', fs=fs)
wb, hb = signal.freqz(bb, ab, 2048, fs=fs)
hbd = 20 * np.log10(np.abs(hb))

# filtr Czebyszewa I
N, fn = signal.cheb1ord(fp, fz, Rp, Rs, fs=fs)
bc1, ac1 = signal.cheby1(N, Rp, fn, 'low', fs=fs)
wc1, hc1 = signal.freqz(bc1, ac1, 2048, fs=fs)
hc1d = 20 * np.log10(np.abs(hc1))

# filtr Czebyszewa II
N, fn = signal.cheb2ord(fp, fz, Rp, Rs, fs=fs)
bc2, ac2 = signal.cheby2(N, Rs, fn, 'low', fs=fs)
wc2, hc2 = signal.freqz(bc2, ac2, 2048, fs=fs)
hc2d = 20 * np.log10(np.abs(hc2))


# filtr Eliptyczny
N, fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)
be, ae = signal.ellip(N, Rp,Rs, fn, 'low', fs=fs)
we, he = signal.freqz(be, ae, 2048, fs=fs)
hed = 20 * np.log10(np.abs(he))


# wszystko razem
ax = plot_response(wb, hbd, None, 'Butterworth')
ax.plot(wc1, hc1d, label='Czebyszew I')
ax.plot(wc2, hc2d, label='Czebyszew II')
ax.plot(we, hed, label='Eliptyczny')
ax.legend()

```
### Opóźnienie grupowe
Każdy filtr wprowadza opóźnienie sygnału - mierzone jako przysunięcie fazowe, lub opóźnienie grupowe, które jest zależne m.in od rzędu oraz typu filtra
``` python
# charakterystyka fazowa
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 6))
for a in ax.ravel():
    a.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    a.grid()
    a.set_xlim(0, 8000)
    a.set_ylim(-180, 180)
ax[0][0].plot(wb, np.degrees(np.angle(hb)))
ax[0][0].set_title('Butterworth')
ax[0][1].plot(wc1, np.degrees(np.angle(hc1)))
ax[0][1].set_title('Czebyszew I')
ax[1][0].plot(wc2, np.degrees(np.angle(hc2)))
ax[1][0].set_title('Czebyszew II')
ax[1][1].plot(we, np.degrees(np.angle(he)))
ax[1][1].set_title('Eliptyczny')
for a in ax[1]:
    a.set_xlabel('Częstotliwość [Hz]')
for a in (ax[0][0], ax[1][0]):
    a.set_ylabel('Faza [°]')

# opóźnienie grupowe
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 6))
for a in ax.ravel():
    a.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    a.grid()
    a.set_xlim(0, 8000)
ax[0][0].plot(*signal.group_delay((bb, ab), 2048, fs=fs))
ax[0][0].set_title('Butterworth')
ax[0][1].plot(*signal.group_delay((bc1, ac1), 2048, fs=fs))
ax[0][1].set_title('Czebyszew I')
ax[1][0].plot(*signal.group_delay((bc2, ac2), 2048, fs=fs))
ax[1][0].set_title('Czebyszew II')
ax[1][1].plot(*signal.group_delay((be, ae), 2048, fs=fs))
ax[1][1].set_title('Eliptyczny')
for a in ax[1]:
    a.set_xlabel('Częstotliwość [Hz]')
for a in (ax[0][0], ax[1][0]):
    a.set_ylabel('Opóźnienie [próbki]')

plt.show()
```

### Filtracja i filtracja zerofazowa
Opóźnienie grupowe  może być obserwowane w sygnale po filtracji:
``` python
### filtracja sygnału
t = np.arange(500)
x1 = np.sin(2 * np.pi * t * 500 / fs)
x2 = np.sin(2 * np.pi * t * 2500 / fs) * 0.5
x = x1 + x2
# sygnały po filtracji
y1 = signal.lfilter(be, ae, x1)
y2 = signal.lfilter(be, ae, x2)
y = signal.lfilter(be, ae, x)

fig, ax = plt.subplots(2, figsize=(8, 6), tight_layout=True)
ax[0].plot(x1)
ax[0].plot(y1)
ax[1].plot(x2)
ax[1].plot(y2)
for aa in ax:
    aa.grid()
    aa.set_ylabel('Amplituda')
ax[1].set_xlabel('Nr próbki')
ax[0].set_title('Sygnał 500Hz przed i po filtracji')
ax[1].set_title('Sygnał 2500Hz przed i po filtracji')
```

jeśli sygnał zostanie przefiltrowany dwukrotnie (od początku i od końca), wtedy opóźnienie grupowe (przesunięcie fazowe) jest równe zero. 
``` python
# filtracja zerofazowa
y2 = signal.filtfilt(be, ae, x)

fig, ax = plt.subplots(2, figsize=(8, 6), tight_layout=True)
ax[0].plot(x)
ax[0].plot(y)
ax[1].plot(x)
ax[1].plot(y2)
for aa in ax:
    aa.grid()
    aa.set_ylabel('Amplituda')
ax[1].set_xlabel('Nr próbki')
ax[0].set_title('Sygnał przed i po filtracji')
ax[1].set_title('Sygnał przed i po filtracji zerofazowej')

```

### Stabilność filtrów IIR
W przypadku stosowania filtrów wyższych rzędów, filtry IIR mogą być niestabilne z uwagi kumulowanie się błędów numerycznych, w wyniku których realne wartości biegunów będą leżały poza okręgiem jednostkowym. W związku z tym należy je reprezentować jako kaskade filtrów rzędu 2 (SOS).
``` python
fs = 48000

x = np.zeros(200, dtype=np.float64)
x[0] = 1

# filtr DP 3 kHz eliptyczny, rzędu 20
b, a = signal.ellip(20,  1, 60, 3000, btype='low',   fs=fs)

y1 = signal.lfilter(b, a, x)
fig, ax = plt.subplots(2, tight_layout=True, figsize=(8, 6))
ax[0].plot(y1)

z, p, k = signal.tf2zpk(b, a)
print(np.max(np.abs(p)))

# filtr SOS
sos = signal.ellip(20,  1, 60, 3000, btype='low',   fs=fs, output='sos')
y2 = signal.sosfilt(sos, x)


stem = ax[1].plot(y2)
plt.show()
```

## Zadania
1. Dla sygnału fs=48kHz, zaprojektuj 4 filtry Czebyszewa 1  rzędu 4: górno przepustowy (fp=3kHz), dolno-przepustowy (fp=3kHz), pasmowo przepustowy (1-3kHz), pasmowo zaporowy (1-3kHz), gdzie tłumienie w paśmie przepustowym nie jest większe niż 1dB. Wyświetl charakterystyki fazowe oraz odpowiedzi impulsowe. 

2. Zakładając, że \\(f_s\\)=500Hz, a okres obseracji wynosi T=2s przygotować wektor sygnału testowego o postaci:

$$
x(t)=sin(3\cdot 2\pi\cdot t)+cos(10\cdot 2\pi\cdot t)+cos(25\cdot 2\pi\cdot t)+sin(35\cdot 2\pi\cdot t)+sin(50\cdot 2\pi\cdot t)+sin(100\cdot 2\pi\cdot t)
$$

3. Zaprojektować filtr dolnoprzepustowy rekursywny Czebyszewa II, który umożliwi stłumienie składowych 50 i 100Hz. Przyjąć, że tłumienie w paśmie zaporowym nie powinno być mniejsze niż 55dB, zaś oscylacje w paśmie przepustowym nie powinny przekraczać 0.1%. Szerokość pasma przejściowego ustalić na 5Hz. Po zaprojektowaniu przefiltruj sygnał i wyznacz opóźnienie filtra dla składowych w paśmie przepustowym.
   
4. Zaprojektuj filtr Eliptyczny pasmowo-zaporowy dla składowej 25Hz, załóż, że pasmo zaporowe ma szerokość 1Hz, a pasmo przejściowe szerokośc taką jak w pkt 3. Określ jakie jest efektywne tłumienie składowej 25Hz sygnału oraz czas ustalania odpowiedzi. 
5. Dokonaj porównania zachowania filtra z zadania 4 z filtrem pasmowo-zaporowym z FIR z zadania 6 z [lab nr 3](http://jug.put.poznan.pl/lab-icmwr/Lab%2003%20-%20Projektowanie%20filtrow%20FIR.html)?
   - Czy czas ustalania się odpowiedzi tego filtra jest dłuższy niż filtra IIR?
   - Czy czas pojawienia się odpowiedzi tego filtra jest dłuższy niż filtra IIR? (jako czas pojawienia się odpowiedzi dla filtra FIR przyjmij moment pojawienia się sygnału na wyjściu)
   - Dla ktorego z filtrów opóźnienie fazowe składowej 25Hz jest większe?
   
   
6. Określ jaka jest minimalna szerokość pasma przejściowego filtra 25Hz z zad 4, która spowoduje, że stanie się on niestabilny.

Autorzy: *Piotr Kaczmarek*
