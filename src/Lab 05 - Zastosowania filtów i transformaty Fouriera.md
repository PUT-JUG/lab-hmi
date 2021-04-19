<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Zastosowania filtrów i transformaty Fouriera

## Zadania
Wszystkie zadania powinny być przesłane w formie pojedynczego pliku o nazwie `cw5.py`. Wszystkie funkcje oraz ich nazwy argumentów powinny być identyczne z przykładami użycia. Dla wszystkich filtrów przyjmij że w paśmie przepustowym oscylacje sygnału nie powinny zmieniać się o więcej niż o 1%.
Wszystkie dane wejściowe zawierająca tablice danych powinny być typu `nd.array`

1. Wczytaj [sygnał testowy](https://chmura.put.poznan.pl/s/kI0ylA5EJBNNcH5)zawierający zarejestrowaną aktywność mięśnia (EMG) brzuchatego łydki. Częstotliwość próbkowania sygnału wynosi \\(f_s\\)=500Hz. Sygnał EMG jest sygnałem, który zawiera składowe w paśmie 15-5000Hz. Przeanalizuje sygnał i napisz funkcję 
``` python
   signal_filtered, signal_filtered_zero_ph  = filter_emg(signal, fs=500, Rs=50, notch=True)
````
Funkcja filtrująca powinna:
- usunąć artefakty ruchowe bez zmiany kształtu sygnału, gdzie tłumienie powinno wynosić nie mniej niż `Rs` [dB]
- usunąć zakłócenie sieciowe i harmoniczne, gdzie oczekiwane stłumienie składowej (i przecieków) zakłócenia harmonicznego nie powinno być mniejsze niż -20dB (na tej podstawie proszę dobrać szerokość pasma)

Funkcja powinna zwracać:
- sygnał po filtracji: `signal_filtered`, 
- sygnał po filtracji zerofazowej: `signal_filtered_zero_ph`

2. Dla przefiltrowanego sygnału napisz funkcję, która dokona subsamplingu sygnału o `r` razy (r jest typu int). Pamiętaj, żeby w przefiltrowanym sygnale nie było aliasów

``` python
   signal_subsampled = subsample_emg(signal_filtered, fs=500, r=3, Rs=50)
````
Filtracja antyaliasingowa nie powinna istotnie zmieniać kształtu sygnału, oraz zapewniać, że ew aliasy będą stłumione o nie mniej niż o `Rs` (wyrażone w dB)

3. Wczytaj [sygnał](https://chmura.put.poznan.pl/s/HoUsRZAWdTviYgl), który zawiera sygnał  siły skurczu mięśnia wywołanej stymulacją elektryczną. Impulsy stymulacji elektrycznej 50-200us i amplitudzie do 200V przenoszą się do ukłądu pomiarowego siły, i tworzą w zapisie charakterystyczne piki. Ponadto w przebiegu widoczny jest szum kwantyzacji przetwornika ADC.
Zastanów się jak stosując filtrację można zmniejszyć amplitudę pików oraz wyeliminować szum kwantyzacji, nie modyfikując kształtu zarejestrowanego sygnału.
Częstotliwość próbkowania sygnału fs=5kHz
Należy wiedzieć że widmo wąskiego impulsu jest zbliżone do widma delty Diraca.


``` python
   signal_filtered, signal_zero_ph = filter_force(signal, fs)
````


Autorzy: *Piotr Kaczmarek*
