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

# Projektowanie filtra metodą okien
- Projektujemy idealną charakterystykę w dziedzinie częstotliwości
- obliczamy odpowiedź impulsową (IFFT)
- przycinamy filtr do żądanej długości za pomocą funkcji okna
- przesuwamy w osi czasu, tak by zaczynał się w t=0

**UWAGA:** Długość filtru, najbardziej uniwersalne są filtry o długości nieparzystej (I typu) (można zrealizować wszystkie typy)


## Zadania
1. Wykorzystując odpowiedź impulsową filtra zaprojektowaną w poprzednim tygodniu zastosuj okno kaisera przycinając filtr do długości 1s (jako długość wybierz najbliższą, nieparzystą liczbę próbek)
2. Przetestuj działanie takiego filtra stosując funkcję `lfilter`przyjmując że argument `b` zawiera współczynniki odpowiedzi impulsowej a `a` wynosi 1. Sprawdź efekt filtracji wyznaczając widmo sygnału
3. Przystosuj metodę do pracy w trybie on-line, tzn. do wielokrotnego wywołania funkcji `lfilter` dla nowych próbek. W takiej sytuacji konieczne jest przechowywanie informacji o stanie filtra. W tym celu wykorzystaj funkcję `lfilter_zi` do wygenerowania stanu początkowego:
`zi = lfilter_zi(b, 1)`

i następnie przy każdym wywołaniu funkcji lfilter:
`out, zi = lfilter(b, a, x,  zi=zi)`

w ten sposób kolejne dane będą dostarczane z uwzględnieniem aktualnego stanu wewnętrznego filtra (informacji o poprzednich próbkach)
4. Spróbuj zaprojektować filtr sieciowy dla systemu Delsys Trigno (fs=50Hz), załóż że szerokość pasma filtracji wynosi +/-1Hz, długość okna 501 próbek
``` python
df = 1
fc = 50
fs = 2000
d = fc - df
fg = fc + df
Nn = 501
b = sig.firwin(Nn, (fd, fg), pass_zero='bandstop', fs=fs)
```
Zastosuj ten filtr do filtracji online danych z czujnika Trigno

Autorzy: *Piotr Kaczmarek*
