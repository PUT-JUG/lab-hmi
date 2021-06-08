<!-- for math equations - MathJax -->
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# Lab11. Analiza zjawiska Potencjałów wywołanych stanu ustalonego w sygnale EEG

## Wprowadzenie
Dzisiejsze zajęcia poświęcone są zapoznaniu się z metodą wykrywania w sygnale EEG obecności bodźca w postaci stałej częstotliwości stymulującej. Rozwiązanie takie umożliwia budowanie interfejsów, gdzie użytkownik koncentruje się na bodźcu o stałej częstotliwości. Wyróżniamy następujące typu bodźców:
- Wzrokowe potencjały wywołane stanu ustalonego (SSVEP), generowane przez migający bodziec
- Somatosensoryczne (czuciowe) potencjały wywołane stanu ustalonego (SEP)
- Słuchowe potencjały wywołane (AEP)

Cechą wspólną interfejsów bazujących na tym zjawisku jest obecność kilku bodźców o różnej, stałej częstotliwości. Wykrycie tej częstotliwości w sygnale EEG w obrębie ośrodka kory odpowiedzialnego za przetwarzanie bodźców z tego ośrodka, jest traktowane jako wyznacznik, tego, że dana osoba koncentruje się na tym bodźcu, lub np. w przypadku systemu do obiektywnej oceny słuchu i czucia oznacza, że bodziec o danym natężeniu dociera i aktywuje dany ośrodek CUN. 

Zadanie detekcji nie jest trywialne z uwagi na dość znaczną zmienność aktywności kory mózgowej, stąd w celu osiągnięcia istotnych statystycznie wyników, obserwacje trzeba prowadzić w horyzoncie czasu dłuższym niż. 1s.

Podczas zajęć przedstawiono sposób wyznaczenia widmo częstotliwości i określenia ilościowego stosunku sygnału do szumu (SNR) przy docelowej częstotliwości w danych EEG zarejestrowanych podczas szybkiej okresowej stymulacji wzrokowej (FPVS) przy 12 Hz i 15 Hz w różnych próbach. Ekstrakcja SNR przy częstotliwości stymulacji jest prostym sposobem ilościowego określenia odpowiedzi oznaczonych częstotliwością w EEG.

Ogólne wprowadzenie do metody patrz Norcia i in. (2015) <https://doi.org/10.1167/15.6.4>_ dla domeny wizualnej oraz Picton i in. (2003) <https://doi.org/10.3109/14992020309101316>_ dla domeny słuchowej.

## Dane

Używamy prostego przykładowego zestawu danych ze stymulacją wizualną ze znacznikami częstotliwości: N=2 uczestników zaobserwowało odwracanie wzorów szachownicy ze stałą częstotliwością 12,0 Hz lub 15,0 Hz. Zarejestrowano 32 kanały  EEG z wykorzystaniem żelu.

Zwizualizujemy zarówno gęstość widmową mocy (PSD), jak i widmo SNR danych z fazy (epoki) stymulacji daną częstotliwością, wyodrębnimy SNR przy częstotliwości stymulacji, wykreślimy topografię odpowiedzi i statystycznie oddzielimy odpowiedzi 12 Hz i 15 Hz w różnych próbach. Ponieważ wywołana odpowiedź jest generowana głównie we wczesnych wizualnych obszarach mózgu, analiza statystyczna zostanie przeprowadzona na potylicznym obszarze ROI.

W przetwarzaniu wykorzystywany będzie moduł `nme`
W celu zainstalowania potrzebnych modułów użyj `pip`
``` python
pip install mne
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

przedstawione rozwiązanie w dużej części jest stworzone przez D. Welke i E. Kalenkovicha:
``` python
# Authors: Dominik Welke <dominik.welke@web.de>
#          Evgenii Kalenkovich <e.kalenkovich@gmail.com>
#
# License: BSD (3-clause)
```

## Wczytane i przetworzenie danych
Sygnał SSVEP ma stosunkowo wysoki stosunek sygnału do szumu (SNR), stąd zaobserwowanie zjawiska nie wymaga stosowania skomplikowanego preprocesingu. Mimo wszystko preprocesing, może poprawić jakość danych. W poniższym przykłądzie preprocessing obejmuje:
- Wczytanie surowego sygnału
- o ile sygnał syrowy (`raw`) zawiera potencjał wyznaczony względem  odprowadzenia FCz, to preprocessing obejmuje wyznaczenie potencjału względem średniej ze wszystkich kanałów, tan zabiego jest stosowany również w EMG i pozwala na usunięcia skłądowej wspólnej (w tym zakłócenia sieciowego)
- filtracja górnoprzepustowa fp=0.1Hz  
- podział przebiegu na 20s części (epoki), w których generowany były bodziec o zadanej częstotliwości
``` python
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import ttest_rel

# Load raw data
data_path = mne.datasets.ssvep.data_path()
bids_fname = data_path + '/sub-02/ses-01/eeg/sub-02_ses-01_task-ssvep_eeg.vhdr'

raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
raw.info['line_freq'] = 50.

# Set montage
montage = mne.channels.make_standard_montage('easycap-M1')
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference('average', projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

# Construct epochs
event_id = {
    '12hz': 255,
    '15hz': 155
}
events, _ = mne.events_from_annotations(raw, verbose=False)
raw.info["events"] = events
tmin, tmax = -1., 20.  # in s
baseline = None
epochs = mne.Epochs(
    raw, events=events,
    event_id=[event_id['12hz'], event_id['15hz']], tmin=tmin,
    tmax=tmax, baseline=baseline, verbose=False)
```

## Analiza czestotliwościowa
Teraz obliczamy widmo częstotliwości danych EEG. Bez konieczności dalszego przetwarzania  dla częstotliwościach stymulacji i niektórych ich harmonicznych,  można zaobserwować piki.

„Klasyczny” wykres PSD zostanie porównany z wykresem widma SNR. SNR będzie obliczany jako stosunek mocy w danym przedziale częstotliwości do średniej mocy w sąsiednich przedziałach. Ta procedura ma dwie zalety w porównaniu z surowym PSD:

 - normalizuje widmo i uwzględnia zanik mocy 1/f.

- modulacje mocy, które mają szerokie pasmo -  znikną.
### Wyznaczenie gęstość widmowej mocy (PSD)
Widmo częstotliwości zostanie obliczone przy użyciu szybkiej transformacji Fouriera (FFT). Jest to powszechna praktyka w literaturze dotyczącej stanu stacjonarnego i opiera się na dokładnej znajomości bodźca i zakładanej reakcji – zwłaszcza pod względem jego stabilności w czasie.

Z analizy wykluczymy pierwszą sekundę każdej próby:

Ustabilizowanie odpowiedzi w stanie ustalonym często zajmuje trochę czasu, a faza przejściowa na początku może zniekształcić oszacowanie sygnału.

oczekuje się, że ta część danych będzie zdominowana przez odpowiedzi związane z pojawieniem się bodźca i co nie jest wykorzystywane w badaniu potencjałów stanu ustalonego.

W module `mne` zwykład transformata FFT jest szczególnym przypadkiem metody Welcha, z tylko jednym oknem Welcha obejmującym całą próbę i bez określonej funkcji okienkowania (zastosowanie okna prostokątnego).

``` python
tmin = 1.
tmax = 20.
fmin = 1.
fmax = 90.
sfreq = epochs.info['sfreq']

psds, freqs = mne.time_frequency.psd_welch(
    epochs,
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    window='boxcar',
    verbose=False)
```
### Wyznaczenie SNR
SNR - tutaj oznacza miarę mocy względnej: jest to stosunek mocy w danym przedziale częstotliwości - "sygnału" - do wartości bazowej "szumu" - średniej mocy w otaczających przedziałach częstotliwości. To podejście zostało pierwotnie zaproponowane przez Meigena i Bacha (1999) <https://doi.org/10.1023/A:1002097208337>_

Należy przyjąć 2 hyperparametry tj: - ile sąsiednich binów częstotliwości należy wziąć do wyznaczenia wartości bazowej szumu i czy chcemy pominąć bezpośrednich sąsiadów (może to mieć sens, jeśli częstotliwość stymulacji nie ma dużej stabilności lub pasma częstotliwości są bardzo wąski). Przeanalizuj poniższą funkcję:

``` python
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise

```
Teraz wywołujemy funkcję, aby obliczyć nasze widmo SNR.
``` python 
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                    noise_skip_neighbor_freqs=1)
```

Poprawne dobranie parametrów jest kluczowe, w dlaszej części będziesz próbował dobrać parametry maksymalizujące waertość SNR dla bodzców.

W aktualnym przykładzie porównywana jest moc w każdym binie ze średnią mocą trzech sąsiednich binach (z każdej strony) i pomijany jest jeden bin bezpośrednio sąsiadujących.

## Wizualizacja PSD i SNR
Wizualizacja przedstawia wyniki z naniesionym obszarem ograniczonym przez wartość +/- std wyznaczone dla wszystkich epok.
``` python
fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))
freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                   np.where(np.ceil(freqs) == fmax - 1)[0][0])

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color='b')
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
    color='b', alpha=.2)
axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color='r')
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
    color='r', alpha=.2)
axes[1].set(
    title="SNR spectrum", xlabel='Frequency [Hz]',
    ylabel='SNR', ylim=[-2, 30], xlim=[fmin, fmax])
fig.show()
```
Przeanalizuj otrzymane wykresy i spróbuj odpowiedzieć na pytania w Quzie
## Wyodrębnienie SNR przy określonej częstotliwości stymulacji
W interfejsach BCI opartych o SSVEP, konieczna jest możliwość odróżnienia częstotliwości bodźca obserwowanego przeze użytkownika. Odróżnienie takie może polegać na analizie statstycznej np. prawdopodobieństwa, że osoba obserwuje dany bodziec. Prawdopodobieństwo może być określone na podstawie rozkąłdów wartości SNR w otoczeniu częstotliwości bodźców.

Poniższsy kod dotyczy częstotliwości 12Hz i umożliwia wyznaczenie binu częstotliwości zawierająceg daną składową oraz ograniczenia liczby kanalóœ do obszaru kory wzrokowej
``` python
# define stimulation frequency
stim_freq = 12.
# find index of frequency bin closest to stimulation frequency
i_trial_12hz = np.where(epochs.events[:, 2] == event_id['12hz'])[0]
i_trial_15hz = np.where(epochs.events[:, 2] == event_id['15hz'])[0]

i_bin_12hz = np.argmin(abs(freqs - stim_freq))
# could be updated to support multiple frequencies

# for later, we will already find the 15 Hz bin and the 1st and 2nd harmonic
# for both.
i_bin_24hz = np.argmin(abs(freqs - 24))
i_bin_36hz = np.argmin(abs(freqs - 36))
i_bin_15hz = np.argmin(abs(freqs - 15))
i_bin_30hz = np.argmin(abs(freqs - 30))
i_bin_45hz = np.argmin(abs(freqs - 45))

# Define different ROIs
roi_vis = ['POz', 'Oz', 'O1', 'O2', 'PO3', 'PO4', 'PO7',
           'PO8', 'PO9', 'PO10', 'O9', 'O10']  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(epochs.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_vis)

snrs_target = snrs[i_trial_12hz, :, i_bin_12hz][:, picks_roi_vis]
print("sub 2, 12 Hz trials, SNR at 12 Hz")
print(f'average SNR (occipital ROI): {snrs_target.mean()}')
```

## Topografia aktywności SSVEP
``` python
# get average SNR at 12 Hz for ALL channels
snrs_12hz = snrs[i_trial_12hz, :, i_bin_12hz]
snrs_12hz_chaverage = snrs_12hz.mean(axis=0)

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_12hz_chaverage, epochs.info, vmin=1., axes=ax)

print("sub 2, 12 Hz trials, SNR at 12 Hz")
print("average SNR (all channels): %f" % snrs_12hz_chaverage.mean())
print("average SNR (occipital ROI): %f" % snrs_target.mean())

tstat_roi_vs_scalp = \
    ttest_rel(snrs_target.mean(axis=1), snrs_12hz.mean(axis=1))
print("12 Hz SNR in occipital ROI is significantly larger than 12 Hz SNR over "
      "all channels: t = %.3f, p = %f" % tstat_roi_vs_scalp)
```
W powyższsym kodzie poza wyświetleneim aktywności poszczególnych obszarów analizowana jest, za pomocą t-testu, różnica między średnią aktywnością wszystkich kanałow a kanałów rozmieszczonych w obrębie kory wzrokowej. Co oznaczają uzyskane wyniki testu?

## Statystyczna rozróżnialność bodźców o częstotliwości 12 i 15Hz
W poprzednich przykładach, wszstkie próby (epoki) były pomieszane, tutaj analizuemy możliwośc rozróżnenie opok, w któryc obserwowana była konkretna częstotliwość.. Wydzielono waerości SNR dla częstotliwości 12 i 15 Hz SNR w obu typach prób (różnej faktycznej cżęstotliwości stymulacji) i porównano wartości za pomocą testu t-studenta. Wyodrębniono również SNR pierwszej i drugiej harmonicznej dla obu częstotliwości stymulacji. Są one również często obserwowane i mogą zwniększyć pewnośc wnioskowania.

``` python
snrs_roi = snrs[:, picks_roi_vis, :].mean(axis=1)

freq_plot = [12, 15, 24, 30, 36, 45]
color_plot = [
    'darkblue', 'darkgreen', 'mediumblue', 'green', 'blue', 'seagreen'
]
xpos_plot = [-5. / 12, -3. / 12, -1. / 12, 1. / 12, 3. / 12, 5. / 12]
fig, ax = plt.subplots()
labels = ['12 Hz trials', '15 Hz trials']
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
res = dict()

# loop to plot SNRs at stimulation frequencies and harmonics
for i, f in enumerate(freq_plot):
    # extract snrs
    stim_12hz_tmp = \
        snrs_roi[i_trial_12hz, np.argmin(abs(freqs - f))]
    stim_15hz_tmp = \
        snrs_roi[i_trial_15hz, np.argmin(abs(freqs - f))]
    SNR_tmp = [stim_12hz_tmp.mean(), stim_15hz_tmp.mean()]
    # plot (with std)
    ax.bar(
        x + width * xpos_plot[i], SNR_tmp, width / len(freq_plot),
        yerr=np.std(SNR_tmp),
        label='%i Hz SNR' % f, color=color_plot[i])
    # store results for statistical comparison
    res['stim_12hz_snrs_%ihz' % f] = stim_12hz_tmp
    res['stim_15hz_snrs_%ihz' % f] = stim_15hz_tmp

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SNR')
ax.set_title('Average SNR at target frequencies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(['%i Hz' % f for f in freq_plot], title='SNR at:')
ax.set_ylim([0, 70])
ax.axhline(1, ls='--', c='r')
fig.show()
```
Można zaobserwować wyraźną różnicę między wartością SNR dla epok o różnej częstotliwości bodźca.

Przeanalizuj zawartośc tablicy res oraz przykład kiedy t-test był wykonywany dla porwnania istotności statystycznej różnicy między obszarem kory wzrokowej a pśrednią aktywnością i spróbuj wyonać t-test, który pozwoli określić czy istnieje istotnie statystyczna różnicą między SNR obserwowanym dla bodźca o częstotliwości 12 i 15 Hz a czestotliwościa 15 i 12Hz. Wyznacz wartość statystyki testowej

## Wpływ długości okresu obserwacji
Najpierw zasymulujemy krótsze próby, biorąc tylko pierwsze x s z naszych dwudziestych prób (2, 4, 6, 8, ..., 20 s) i obliczymy SNR za pomocą okna FFT, które obejmuje całą epokę:
``` python
stim_bandwidth = .5

# shorten data and welch window
window_lengths = [i for i in range(2, 21, 2)]
window_snrs = [[]] * len(window_lengths)
for i_win, win in enumerate(window_lengths):
    # compute spectrogram
    windowed_psd, windowed_freqs = mne.time_frequency.psd_welch(
        epochs[str(event_id['12hz'])],
        n_fft=int(sfreq * win),
        n_overlap=0, n_per_seg=None,
        tmin=0, tmax=win,
        window='boxcar',
        fmin=fmin, fmax=fmax, verbose=False)
    # define a bandwidth of 1 Hz around stimfreq for SNR computation
    bin_width = windowed_freqs[1] - windowed_freqs[0]Rozwią
    skip_neighbor_freqs = \
        round((stim_bandwidth / 2) / bin_width - bin_width / 2. - .5) if (
            bin_width < stim_bandwidth) else 0
    n_neighbor_freqs = \
        int((sum((windowed_freqs <= 13) & (windowed_freqs >= 11)
                 ) - 1 - 2 * skip_neighbor_freqs) / 2)
    # compute snr
    windowed_snrs = \
        snr_spectrum(
            windowed_psd,
            noise_n_neighbor_freqs=int(n_neighbor_freqs) if (
                n_neighbor_freqs > 0
            ) else 1,
            noise_skip_neighbor_freqs=int(skip_neighbor_freqs))
    window_snrs[i_win] = \
        windowed_snrs[
        :, picks_roi_vis,
        np.argmin(
            abs(windowed_freqs - 12.))].mean(axis=1)

fig, ax = plt.subplots(1)
ax.boxplot(window_snrs, labels=window_lengths, vert=True)
ax.set(title='Effect of trial duration on 12 Hz SNR',
       ylabel='Average SNR', xlabel='Trial duration [s]')
ax.axhline(1, ls='--', c='r')
fig.show()
```
Przeanalizuje otrzymany wykres.

## Zadanie
1. Wykonaj zadania z wprowadzenia i przeanalizuj otrzymane wykresy i wyniki testów statystycznych
2. Przeanaliuj wyniki dla topografii i istnienia istotnej statsytycznie , wyższej aktywności obszaru korowego dla częstotliwości 15Hz
3. Przeanaliuj wpływ długości okna na SNR dla częstotliwości 15Hz
4. Odpowiedz na ptania z Quizu na plarformie ekursy



 