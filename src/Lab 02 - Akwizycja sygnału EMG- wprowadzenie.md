<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Transformata Fouriera sygnałów dyskretnych- splot, modulacja, okna

## Wstęp
### Delsys Trigno SDK
Moduły Delsys Trigno zawierają 1 kanałowy rejestrator EMG pracujący z częstotliwością 1259Hz oraz IMU (Akcelerometr +/116g, gyroskop +/-20000dps) pracujący z częstotliwościa 148Hz.
Komunikacja z modułami dobywa się przez serwer (Delsys Trigno Controll Utility).
Do komunikacji z serwerem możesz wykorzystać moduł *pytringos* (git@github.com:biolab-put/pytrignos.git)

przykłady uzycia znajdziesz w katalogu examples.
#### Inicjacja sensora
``` python
trigno_sensors = TrignoAdapter()
trigno_sensors.add_sensors(sensors_mode='EMG', sensors_ids=(4,), sensors_labels=('EMG1',), host='192.168.4.118')
trigno_sensors.add_sensors(sensors_mode='ORIENTATION', sensors_ids=(4,), sensors_labels=('ORIENTATION1',), host='192.168.4.118')
```

`class  TrignoAdapter.add_sensor(sensor_mode, sensor_ids, sensor_labels, host)`
```
    sensors_mode : str
               Rodzaj sensora do odczytu. (tj. 'ORIENTATION' lub 'EMG')
    sensors_ids : tuple
               Identyfikatory wykorzystanych modułów np. (1, 2,) będzie odczytywał dane z modułów 1 i 2
    sensors_labels : tuple
               Etykiety opisujące moduły w danych wyjściowych (kolumna 'Sensor_id') , np. ('ORIENTATION1', 'ORIENTATION2',).
    host : str
                Adres IP serwera Delsys Trigno Controll Utility 
```
#### Akwizycja danych
Dane odczytywane z modułu sa buforowane i dostępne na żądanie. Poniższy kod odczytuje zgromadzone dane co 1s.

``` python
trigno_sensors.start_acquisition()

time_period = 1.0 #s
while(True):
    time.sleep(time_period)
    sensors_reading = trigno_sensors.sensors_reading()
    print(sensors_reading)
trigno_sensors.stop_acquisition()
```

## Przebieg ćwiczenia
1. Uruchom podstawowy kod  z wprowadzenia (patrz również [przykład](https://github.com/biolab-put/pytrignos/blob/main/example/test_reading.py)). Numer IP serwera poda prowadzący, numer sensora jest opisany na obudowie modułu
2. Uruchom [kod](https://github.com/biolab-put/pytrignos/blob/main/example/test_reading_plot.py) umożlwiajacy odczyt danych z czujnika z wykorzystaniem [FuncAnimation](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib.animation.FuncAnimation)
3. Sprawdź czy jesteś w stanie zaobserwować aktywność swoich mięśni przykładając czujnik np. do mięśnia odwodziciela kciuka i modulować napięce tego mięśnia
4. Zarejestruj sygnał i przy wyjściu zapisz go do pliku (w formacie hdf)
5. Zmodyfikuj funkcję wyświetlającą tak, by wyświetlała co 1s wyświetlała widmo sygnału, na osi x nanieś faktyczną częstotliwość
6. Przeanalizuj poniższy kod w którym w celu ograniczenia wycieków widma stosuje się okno Hanna:
``` python
def spect_dB(s, N_fft, F_samp):
    S = fft(s,N_fft)
    S_dB = 20 * np.log10(np.abs(S))
    F = fftfreq(N_fft, 1.0/F_samp)
    return (S_dB,F)

fs=100
f=5

plt.figure()

k=1
for T in np.linspace(1,1.2,4):
    t = np.arange(0,T,1/fs)
    window = signal.windows.hann(len(t))#
    s = np.sin(2*np.pi*f*t)
    s_wnd = s* window
    plt.subplot(4,1,k)
    plt.title(f'T={T}')
    S_wnd, F = spect_dB(s_wnd,len(s), fs)
    S, F = spect_dB(s,len(s), fs)
    plt.plot(F,S_wnd)
    plt.plot(F,S)
    
    k=k+1
```
- Jak zmienia się widmo sygnału bez zastosowania okna Hann'a w zależności od długości sygnału?
   
7. Przed wyznaczeniem widma wymnażaj sygnał EMG przez okno Hanna co zmienia się w widmie w porównaniu sygnałem nie wymnożonym?   
---
Autorzy: *Piotr Kaczmarek*
