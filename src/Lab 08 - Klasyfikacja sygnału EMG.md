<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Podstawy klasyfikacji sygnału EMG (częśc 2)

## Wprowadzenie
Dzisiejsze zajęcia są kontynuacją zajęć dotyczących oceny możliwości zastosowania sygnału EMG do rozpoznawania gestów.

W zadaniu wykorzystaj skrypty parsujące, które można pobrać z [GitHub](https://github.com/biolab-put/putemg_features.git), pamiętaj, żeby sklonować repozytorium razem z submodułami.

Podczas zajęć wykorzystasz cechy używane w zeszłym tygodniu oraz dodatkowo cechy ze zbioru zaproponowanego przez Hudgkina.
Wywołanie kodu generującego cechy:
``` python
df1 = putemg_features.calculate_feature(record.loc[:, columns_emg], 'ZC', window=500, step=250, threshold=30)
df2 = putemg_features.calculate_feature(record.loc[:, columns_emg], name='RMS', window=500, step=250)
df3 = putemg_features.calculate_feature(record.loc[:, columns_emg], name='SSC', window=500, step=250, threshold=16)
df4 = putemg_features.calculate_feature(record.loc[:, columns_emg], name='WL', window=500, step=250)
```
Gdzie `window`, `step` są podane w próbkach a nie w jednostkach mianowanych czasu.

Inną przydatną funkcją, która pozwala na nie uwzględnianie w uczeniu próbek znajdujących się w otoczeniu tranzycji jest funkcja:
``` python
 y_true = putemg_features.biolab_utilities.filter_transitions(y_true.values,
                                                   start_before=2, start_after=1,
                                                   end_before=0, end_after=0,
                                                   pause_before=0, pause_after=4)
```

gdzie `start_before`, `start_after` określają liczbę próbek, która powinna zostać usunięta przed i po tranzycji  nowego gestu.  `end_before`dotyczy końca gestu,  `end_after`,  `pause_before`, `pause_after` dotyczy początku i końca okresu pauzy

## Zadanie
1. Dany jest [zbiór danych](https://chmura.put.poznan.pl/s/iuvJ6vRXAeyiEOh) zawierający zbiór treningowy oraz testowy w wersji surowej i po preprocessingu (filtracji)
2. Wyznacz jego cechy RMS i ZC oraz skojarzone z nimi etykiety ('TRAJ_GT')
3. Przyjmij, że w zadaniu klasyfikacji przetwarzane są wyłącznie próbki, dla których etykieta ma wartość >=0
4. Zakładając, ze zbiór uczący jest wykorzystywany w procesie doboru hiperperametrów modelu oraz wyboru cech, spróbuj wyznaczyć precyzję, oraz niepewność jej szacowania. Dla otrzymanych wyników określ, które ze zmian dają istotną statystycznie poprawę, oraz, który z klasyfikatorów jest najlepszy. Test przeprowadź używając testu one-way Anova. Uwzględnij następujące sytuacje:
   - wprowadzenie sygnału po filtracji (sygnały z postfixem `_pre`) oznacz go jako `raw` i `filtered`
  
   - dodanie nowych cech (w pierwotnej wersji klasyfikator wykorzystuje `RMS` i `ZC`) oznacz je jako `full` i `rms_zc`
   - usuwanie tranzycji z danych uczących, oznacz je jako `trans_excluded`, `trans_included`
   - 3 klasyfikatory LDA (`lda`), MLP (`mlp`), RandomForestClassifier (`rf`)
Wygeneruj wyniki klasyfikacji dla zbioru walidacyjnego, gdzie w jednej kolumnie ('precision') umieszczony jest wynik dokładności klasyfikacji danego wywołania klasyfikatora a w kolumnie drugiej jest etykieta (`pipeline_id`) informująca o sposobie przetworzenia zestawu cech. Np dla klasyfikacji sygnału bez filtracji, z usuniętymi tranzycjami, i ograniczonym zestawem cech dla klasyfikatorea lda etykieta będzie miała wartość `raw_rms_zc_trans_excluded_lda`. 
Pamiętaj, że dla części klasyfikatorów wynik zależy od parametrów początkowych, stąd dla analizy istotoności operację uczenia należy powtórzyć min 20 razy dla różnych parametrów początkowych (`initail_state`).


3.  Jako zadanie domowe prześlij plik źródłowy realizujący uczenie oraz plik `predictions.hdf` zawierający wyniki klasyfikacji dla różnych kombinacji metod. Plik powinien zawierać 2 kolumny `precision` i `pipeline_id` oraz odpowiedz na pytanie w Quizie



 