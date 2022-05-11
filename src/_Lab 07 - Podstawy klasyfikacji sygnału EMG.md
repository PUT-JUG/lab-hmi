<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Podstawy klasyfikacji sygnału EMG

## Wprowadzenie
Dzisiejsze zajęcia dotyczą oceny możliwości zastosowania sygnału EMG do rozpoznawania gestów.
W zadaniu wykorzystaj skrypty parsujące stworzone podczas poprzednich zajęć dotyczących przetwarzania.

## Zadanie
1. Wczytaj sygnał [MVC](https://chmura.put.poznan.pl/s/4UuSx0lfK53FA7I), i sygnał [treningowy](https://chmura.put.poznan.pl/s/38aeyGzigLEHLbp)
2. Wyznacz jego cechy RMS i ZC oraz skojarzone z nimi etykiety ('TRAJ_GT')
3. Przyjmij, że w zadaniu klasyfikacji przetwarzane są wyłącznie próbki, dla których etykieta ma wartość >=0
4. Przeanalizuj działanie poniższego programu, którego celem jest oszacowanie dokładności klasyfikatora metodą [k-krotnej walidacji](https://scikit-learn.org/stable/modules/cross_validation.html)
```
names = [ "Random Forest", "LDA", "Neural Net"]
classifiers = [    
    RandomForestClassifier(n_estimators=100, random_state = 0),
    MLPClassifier(alpha=0.9, max_iter=1000, random_state = 0),
    LinearDiscriminantAnalysis()
   ]

for name, clf in zip(names, classifiers):
      scores = cross_val_score(clf, train_features, train_labels, cv=5,  scoring='precision_macro')
      print(name, ', precision mean:', np.mean(scores), "std: ", np.std(scores))
```
5. Porównaj otrzymane wyniki z wynikami klasyfikacji sygnału [testowego](https://chmura.put.poznan.pl/s/7g3b2p7tljJJaNc) zarejestrowanego w tej samej sesji. Do tego celu wytrenuj klasyfikator na danych uczących a następnie dokonaj predykcji dla zbioru testowego. Do oceny możesz użyć np. funkcji:
``` python
metrics.precision_score(predictions, test_labels, average='macro')
```
pamiętaj o tym, że zastosowany sposób uśredniania (`macro`) jest związany z zróżnicowaną liczebnością grup

6. Dla wyników z zad. 5, sprawdź macierz pomyłek i zastanów które gesty są najczęściej mylone ze sobą, spróbuj wyjaśnić dlaczego?

7. (*) Stosując k-krotną walidację, spróbuj zmodyfikować parametry klasyfikatora żeby zwiększyć jego dokładność. Pamiętaj żeby do oceny postępów nie używać zbioru testowego
   
8.  Pobierz sygnał [walidacyjny](https://chmura.put.poznan.pl/s/wuu8IZDRHxUrgXX) i wyznacz jego cechy. Wygeneruj predykcję gestów dla chwil czasu danych [indeksami](https://chmura.put.poznan.pl/s/3S2QorjXu0tUM0h). Pamiętaj, że jeśli nie będziesz realizował zadania 7* nie zmieniać wartości to żeby nie zmieniać wartości `random_state` klasyfikatora. Do realizacji wybierz klasyfikator o najlepszych właściwościach generalizacyjnych, możesz go przeuczyć zgodnie ze swoimi umiejętnościami. Otrzymany plik z predykcją etykiet zapisz do pliku hdf. Plik powinien mieć indeks odpowiadający indeksowi z pobranego pliku z indeksami oraz kolumnę `predictions`. 
9.  Jako zadanie domowe prześlij plik źródłowy realizujący uczenie oraz plik `predictions.hdf` zawierający wyniki klasyfikacji dla zbioru walidacyjnego.


 