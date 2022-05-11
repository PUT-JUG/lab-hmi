<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Transformata Fouriera sygnałów dyskretnych- splot, modulacja, okna

## Wstęp

### Filtry FIR (Finite Impulse Response)
Jest to klasa filtrów nierekursywnych, czyli nie posiadających sprzężeń zwrotnych. Oznacza to że odpowiedź filtru wyznaczana jest wyłącznie jako kombinacja liniowa pewnej ilości próbek sygnału wejściowego.
$$
y(n)=\sum^{\infty}_{m=-\infty}h(m)x(n-m)
$$
gdzie \\(x(n)\\) jest sygnałem wejściowym, \\(y(n)\\) sygnałem wyjściowym, zaś \((h(n)\\) odpowiedzią impulsową filtra.

### Cechy filtrów FIR

1. Z uwagi na skończoną długość filtra jego odpowiedź na jest zawsze stabilna.
2. \\(lim_{n\rightarrow\infty}y(n)=0\\).
3. Filtr może być zaprojektowany tak by mieć liniową charakterystykę fazową. Właściwość ta gwarantuje, że filtrowany sygnał nie będzie zniekształcany (będzie miał stałe opóźnienie grupowe)

### Właściwości filtrów z liniową charakterystyką fazową
W ćwiczeniu ograniczymy się do jednego typu filtru mającego następujące właściwości:

1. długość filtru \\(N=2L+1\\) - filtr ma nieparzystą liczbę próbek.
2. pulsacja unormowana: \\(\omega \in <0;2\pi\\).
3. charakterystyka amplitudowa filtru jest symetryczny względem osi \\(0, \pi\\).
4. charakterystyka częstotliwościowa filtru posiada wyłącznie część rzeczywistą (\\(Im(H(s))=0\\)).
5. charakterystyka czasowa filtru jest również symetryczna \\(h(n)=h(N-1-n)\\) warunek ten jest gwarantowany pośrednio przez symetrię charakterystyki w dziedzinie częstotliwości.

Filtr posiadający takie właściwości ma również liniową charakterystykę fazową (stałe opóźnienie grupowe). W tej klasie mogą istnieć filtry: LP, HP, BP, BS.

### Okna i ich właściwości
Skończoną długość odpowiedzi impulsowej obiektu wykorzystywanego w procesie filtracji można opisać jako wycięcie nieskończonej odpowiedzi impulsowej filtru idealnego przez pewne okno.
Szerokość oraz parametry okna determinują ostateczną jakość filtracji (szerokość pasma przejściowego oraz tłumienie w paśmie zaporowym).
Z uwagi na ``listki boczne`` w charakterystyce amplitudowej okna, użycie każdego z okien musi doprowadzić do ograniczenia tłumienia w paśmie zaporowym, ponadto z uwagi na niezerową szerokość listka głównego zmniejsza się stromość charakterystyki w paśmie przejściowym.
Dalej zostały opisane 3 okna oraz ich właściwości. 

Przyjęto następujące oznaczenia:

 - \\(w(n)\\) - funkcja okno,
 - \\(N=2M+1\\) - jest nieparzystą(!) długością okna,
- \\(A_{sl}\\) - względny poziom tłumienia ``listków bocznych`` w stosunku do ``listka głównego``,
- \\(\Delta_{ml}=\frac{\Delta f}{f_s}\\) unormowana szerokość ``listka głównego``, gdzie \\(\Delta f\\) jest szerokością pasma przejściowego a \\(f_s\\) jest okresem próbkowania,
- \\(A_{stop}\\) - maksymalne wzmocnienie filtra w paśmie zaporowym.

#### Okno prostokątne

- \\(w(n)= 1 :n\in<-M;M>\\)
- \\(N>\frac{2}{\Delta_{ml}}\\)
- \\(A_{sl}=13,5dB\\)
- \\(A_{stop}=-21dB\\)

Uwaga: Dla filtrów wyciętych za pomocą okna prostokątnego można wyłącznie regulować stromość charakterystyki w paśmie przejściowym.

#### Okno Hanninga
- \\(w(n)= 0,5+0,5cos(2\pi n/(2M+1)) :n\in<-M;M>\\) 
- \\(N>\frac{4}{\Delta_{ml}}\\)
- \\(A_{sl}=31dB\\)
- \\(A_{stop}=-44dB\\)

Uwaga: Dla filtrów wyciętych za pomocą okna Hanninga można wyłącznie regulować stromość charakterystyki w paśmie przejściowym.

#### Okno Kaisera
[Okno Kaisera](https://en.wikipedia.org/wiki/Kaiser_window) w odróżnieniu od poprzednio wspomnianych umożliwia na dostosowanie tłumienia w paśmie zaporowym.
Okno dane jest wzorem:

\\(w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}} \right)/I_0(\beta)\\)
z

\\(\quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},\\)
gdzie \(I_0\) jest zmodyfikowaną funkcją Bessela.

W Pythonie okno tworzone jest za pomocą:

``` python
window = signal.kaiser(N, beta)
```

gdzie `N` jest liczbą próbek a `beta` jest parametrem kształtu określającym pewien kompromis między szerokością listka głównego a tłumieniem dla listów bocznych.

Do wyznaczenia współczynnika beta, gdzie zadane jest tłumienie `ripple` (w dB) zarówno w paśmie przepustowym jak i zaporowym oraz szerokości pasma przejściowego (`width`) wyrażonego w pulsacji znormalizowanej (width=1 odpowiada szerokości pasma przejściowego `fs/2`) można wykorzystać funkcję:
``` python
scipy.signal.kaiserord(ripple,width)
```
Więcej na ten temat możesz znaleźć w [dokumentacji](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html)

### Metody projektowania filtrów
Metody projektowania filtrów zostały przedstawione w materiałach z wykładów. Tutaj przedstawiono wyłączenie przykłady realizacji filtrów w języku Python.
Zwróć uwagę na użycie funkcji `firwin` oraz `lfilter`. Przeanalizuj również działanie funkcji `freqz` umożliwiającej wyświetlenie charakterystyki amplitudowej. Do uruchomienia skryptu wykorzystaj plik [distorted.npy](https://chmura.put.poznan.pl/s/cASE9TVojr2I6CK)

``` python

"""Filtracja zakłóceń z sygnału."""

import numpy as np
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8, 4)

wav = np.load('distorted.npy')
wav = wav / np.max(np.abs(wav))
fs = 48000
t = np.arange(len(wav)) / fs

# oryginalny - postać czasowa
plt.figure()
plt.plot(t[:500], wav[:500])
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title('Sygnał oryginalny - postać czasowa')

# oryginalny - widmo
spectrum = np.fft.rfft(wav * np.hamming(2048))
spdb = 20 * np.log10(np.abs(spectrum) / 1024)
f = np.fft.rfftfreq(2048, 1 / 48000)
plt.figure(figsize=(8, 4))
plt.plot(f, spdb)
ideal = np.zeros_like(spdb)
ideal[f > 3000] = -60
# plt.plot(f, ideal, c='r')
plt.xlim(0, 12000)
plt.ylim(bottom=-60)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Poziom widma [dB]')
plt.title('Sygnał oryginalny - postać widmowa')

# filtr dolnoprzepustowy
fc = 1900  # częstotliwość graniczna
N = 101  # długość filtru
h = sig.firwin(N, fc, pass_zero='lowpass', fs=48000)

# z, p, k = sig.tf2zpk(h, 1)
# print(len(z), len(p))

# odpowiedź impulsowa
plt.figure(figsize=(10, 4))
plt.stem(h, use_line_collection=True)
plt.xlabel('Indeks')
# plt.ylabel('Amplituda')
plt.title('Odpowiedź impulsowa filtru')
# plt.title('Współczynniki h')

# charakterystyka częstotliwościowa
w, hf = sig.freqz(h, worN=2048, fs=48000)
hfdb = 20 * np.log10(np.abs(hf))
phase = np.degrees(np.angle(hf))

plt.figure()
fig1, ax1 = plt.subplots(2, sharex=True, tight_layout=True, figsize=(8, 5))
ax1[0].plot(w, hfdb)
ax1[0].set_xlim(0, 12000)
ax1[0].set_ylim(bottom=-80)
ax1[0].set_xlabel('Częstotliwość [Hz]')
ax1[0].set_ylabel('Poziom widma [dB]')
ax1[0].set_title('Charakterystyka częstotliwościowa')
ax1[0].grid()


ax1[1].axvline(3000, c='k', lw=1, ls='--')
ax1[1].plot(w, phase)
ax1[1].grid()
ax1[1].set_ylim(-180, 180)
ax1[1].set_xlabel('Częstotliwość [Hz]')
ax1[1].set_ylabel('Faza [°]')
ax1[1].set_title('Charakterystyka fazowa filtru DP 3 kHz')


# charakterystyka częstotliwościowa + sygnał
plt.figure()
plt.plot(f, spdb)
plt.plot(w, hfdb)
plt.xlim(0, 12000)
plt.ylim(bottom=-80)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Poziom widma [dB]')
plt.title('Charakterystyka częstotliwościowa')
plt.grid()

# filtracja sygnału
y = sig.lfilter(h, 1, wav)
ysp = np.fft.rfft(y * np.hamming(2048))
yspdb = 20 * np.log10(np.abs(ysp) / 1024)

# widmo sygnału przed i po filtracji
plt.figure()
plt.plot(f, spdb, c='#a0a0a0', label='Oryginalny')
plt.plot(f, yspdb, label='Po filtracji')
plt.xlim(0, 12000)
# plt.ylim(bottom=0)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Poziom widma [dB]')
plt.legend()
plt.grid()
plt.title('Widmo sygnału przed i po filtracji')

# postać czasowa przed i po filtracji
fig7, ax7 = plt.subplots(2, figsize=(8, 5), sharex=True, tight_layout=True)
ax7[0].plot(t[:500], wav[:500], label='Originalny')
ax7[1].plot(t[:500], y[50:550], label='Przetworzony')
ax7[-1].set_xlabel('Czas [s]')
for a in ax7:
    a.set_ylabel('Amplituda')
    a.legend(loc='upper right')
ax7[0].set_title('Sygnał oryginalny i przetworzony - postać czasowa')

plt.show()

```
#### Opóźnienie grupowe
Każdy filtr wprowadza opóźnienie sygnału, które jest zależne m.in od długości filtra
``` python
import numpy as np
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8, 4)


n = 1000
fs = 4800
t = np.arange(n) / fs
fr = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
x = np.sum(np.sin(2 * np.pi * t * fr), axis=0)
x = x + 0.05 * np.random.randn(len(x))
x = x / np.max(np.abs(x))

h = sig.firwin2(801, [0, 1250, 1300, fs/2], [1,1,0,0], window='hamming', fs=fs)
y = sig.lfilter(h, 1, x)

fig, ax = plt.subplots(2, sharex=True, tight_layout=True, figsize=(8, 5))
ax[0].plot(x[:n])
ax[1].plot(y[:n])
for a in ax:
    a.grid()
    a.set_ylabel('Amplituda')
ax[1].set_xlabel('Nr próbki')
ax[0].set_title('Sygnał wejściowy')
ax[1].set_title('Sygnał po filtracji (N=801)')

plt.show()
```

## Zadania
1. Dla sygnału fs=48kHz, zaprojektuj 4 filtry o długości N=101 próbek z oknem Hamminga: górno przepustowy (fp=3kHz), dolno-przepustowy (fp=3kHz), pasmowo przepustowy (1-3kHz), pasmowo zaporowy (1-3kHz). Wyświetl charakterystyki fazowe oraz odpowiedzi impulsowe. 

2. Zakładając, że \\(f_s\\)=500Hz, a okres obseracji wynosi T=2s przygotować wektor sygnału testowego o postaci:

$$
x(t)=sin(3\cdot 2\pi\cdot t)+cos(10\cdot 2\pi\cdot t)+cos(25\cdot 2\pi\cdot t)+sin(35\cdot 2\pi\cdot t)+sin(50\cdot 2\pi\cdot t)+sin(100\cdot 2\pi\cdot t)
$$

3. Zaprojektować filtr dolnoprzepustowy nierekursywny, który umożliwi stłumienie składowych 50 i 100Hz. Dla okna Kaisera przyjąć, że tłumienie w paśmie zaporowym nie powinno być mniejsze niż  55dB, zaś oscylacje w paśmie przepustowym nie powinny przekraczać 0.1%. Szerokość pasma przejściowego ustalić na 5Hz. Do projektowania wykorzystać metodę `firwin` oraz okno Kaisera. Po zaprojektowaniu przefiltruj sygnał i wyznacz opóźnienie filtra.
   
4. Zaprojektuj filtr pasmowo-zaporowy dla składowej 25Hz i 50Hz, załóż że pasmo zaporowe ma szerokość 1Hz, a pasmo przejściowe szerokośc taką jak w pkt 3. Wybierz metodę `firwin2` oraz założenia projektowe analogiczne do tych z pkt 3.
   
5. Zaprojektuj filtry z zadania 3 i 4 metodą najmniejszych kwadratów (`firls`). Jakie jest najsłabsze tłumienie w paśmie zaporowym? 
   
6. W celu porównania skuteczności działania filtra zaprojektowanego metodą `firls` i `firwin` z zadania 4 i 5 wyznacz opóźnienie każdego z filtrów i przesuń sygnał przefiltrowany w taki sposób żeby pokrywał się z sygnałem oryginalnym, bez składowych, które miały zostać usunięte. Następnie w oknie o długości 1s (w przedziale 0.5-1.5s) wyznacz błąd średniokwadratowy liczony jako wartość średnia sumy kwadratów błędu.

Autorzy: *Piotr Kaczmarek*
