# The A-Files — przewodnik po README (wersja polska)

Ten dokument opisuje po polsku strukturę i zawartość głównego pliku [README.md](../README.md). Nie jest jego pełnym
tłumaczeniem: dla każdej sekcji wyjaśnia, czego dotyczy, jakie informacje zawiera i kiedy warto do niej zajrzeć.
Numeracja sekcji odpowiada numeracji w README.

**The A-Files** (`taf`) to otwartoźródłowe narzędzie badawcze do powtarzalnej oceny metod steganografii i znakowania
wodnego (watermarkingu) w sygnałach mowy. Zawiera implementacje referencyjne algorytmów osadzania, obiektywne miary
przezroczystości percepcyjnej i zrozumiałości, sparametryzowany model zniekształceń kanału i ataków oraz procedurę
steganalizy szacującą wykrywalność.

---

## 1. Zakres i sformułowanie problemu ([Scope and problem formulation](../README.md#about))

Sekcja wprowadza problem badawczy. Ukrywanie informacji w audio podlega czterem sprzecznym wymaganiom:
**pojemności** (ile bitów można osadzić), **przezroczystości percepcyjnej** (na ile osadzenie jest niesłyszalne),
**odporności** (czy wiadomość przetrwa przetwarzanie sygnału) oraz — w przypadku steganografii —
**niewykrywalności statystycznej**. Autorzy wskazują, że wyniki publikowane w literaturze są często nieporównywalne, a
narzędzie ma temu zaradzić przez jednolity, w pełni opisany protokół eksperymentalny.

Sekcja definiuje notację używaną w całym dokumencie:

* `x[n]` — sygnał nośny (cover), `b` — binarna wiadomość o długości `L`;
* `y[n] = E(x[n], b)` — sygnał z osadzoną wiadomością (stego);
* `z[n] = A_θ(y[n])` — sygnał po ataku o jawnych parametrach `θ`;
* `b̂ = D(z[n], L)` — wiadomość odtworzona przez ślepy dekoder (bez dostępu do sygnału nośnego).

Każda próba jest opisywana przez: stopę błędów bitowych (BER) i dokładność bitową, przezroczystość (mierzoną *przed*
atakiem, aby nie mieszać zniekształceń osadzenia ze skutkami ataku), odporność (BER w funkcji rodzaju i siły ataku),
pojemność oraz wykrywalność.

Podsekcje:

* **Signal and payload representation** — obsługiwane formaty (WAV, FLAC, OGG) oraz dołączone podzbiory korpusów VCTK
  i LibriSpeech, które pozwalają powtarzać eksperymenty bez pobierania danych.
* **Method contract** — cztery własności weryfikowane automatycznym testem dla każdej metody: bezbłędne odtworzenie
  wiadomości przez *nową* instancję dekodera, brak modyfikacji sygnału wejściowego i jego długości, zgłoszenie
  `ValueError` przy przekroczeniu pojemności oraz stabilność na sygnałach syntetycznych. Wyjaśnia też, dlaczego siły
  osadzenia są definiowane względnie (np. względem normy ramki), a nie jako stałe bezwzględne.

## 2. Instalacja ([Installation](../README.md#install))

Instalacja pakietu z PyPI (`pip install the-a-files`) oraz tabela opcjonalnych rozszerzeń:

* `neural` — wytrenowane sieci do znakowania wodnego (AudioSeal, WavMark; PyTorch);
* `ai` — metoda FGAS i metryka MOSNet (TensorFlow);
* `experiments` — silnik eksperymentów i REST API;
* `dev` — narzędzia do testów i budowania pakietu.

## 3. Użycie ([Usage](../README.md#usage))

Uruchamianie wbudowanego przepływu ewaluacji poleceniem `taf-eval` (scenariusze `direct-no-metrics` i `full` lub własny
plik YAML) oraz przykład bezpośredniego użycia metody przez fabrykę `SteganographyMethodFactory`: osadzenie
i odczyt wiadomości.

## 4. Silnik eksperymentów, REST API i platforma webowa ([Experiment engine, REST API and web platform](../README.md#platform))

* **4.1 Silnik eksperymentów** — deklaratywna definicja eksperymentu (`ExperimentConfig`): zbiór danych, metody,
  metryki, ataki, długości wiadomości, liczba powtórzeń i ziarno losowości. Opisuje sześć typów eksperymentów
  współdzielących jeden format wierszy wynikowych oraz analizy pochodne (macierz odporności, progi pojemności,
  ranking metod).
* **4.2 REST API** — uruchomienie serwera FastAPI i tabela punktów końcowych (katalog, uruchamianie, wyniki, postęp
  przez Server-Sent Events, wgrywanie danych).
* **4.3 Panel webowy** — aplikacja Next.js w katalogu `web/`, będąca cienkim klientem API; cała logika dziedzinowa
  pozostaje w pakiecie Pythona. Panel nie jest częścią dystrybucji PyPI.

## 5. Metody steganografii i znakowania wodnego ([Steganography and watermarking methods](../README.md#steganography-algorithms))

Tabela 1 zawiera 27 zaimplementowanych metod wraz z odwołaniami do publikacji źródłowych. Obejmują one metody
w dziedzinie czasu (LSB, echo, histogram, modyfikacja amplitudy niskich częstotliwości), w dziedzinach transformat
(DCT, DWT, LWT, SVD), rozpraszanie widma (DSSS, Improved Spread Spectrum), modulację indeksem kwantyzacji (QIM),
kodowanie fazy, metody adaptacyjne (AAC + kody STC) oraz metody neuronowe (FGAS, AudioSeal, WavMark).

Sekcja pokazuje też abstrakcyjny interfejs `SteganographyMethod` (`encode`, `decode`, `type`) i przypomina, że nowe
metody trzeba zarejestrować w fabryce oraz w typie `MethodType`, a także spełnić kontrakt z sekcji 1.

## 6. Obiektywne metryki jakości ([Objective quality metrics](../README.md#metrics))

21 metryk porównujących sygnał nośny z sygnałem przetworzonym, pogrupowanych według mierzonej własności
(tabele 2–5, numeracja ciągła):

* **6.1 Metryki oparte na uczeniu maszynowym** — MOSNet, przewidujący średnią ocenę subiektywną (MOS);
* **6.2 Pogłos mowy** — BSD i SRMR;
* **6.3 Zrozumiałość mowy** — CSII, NCM, STOI;
* **6.4 Jakość mowy** — miary wierności sygnału i jakości percepcyjnej: SNR, SNRseg, fwSNRseg, LLR, WSS, odległości
  cepstralne (CD, MCD), PESQ, metryki złożone (Csig, Cbak, Covl), wSTMI, STGI, SI-SDR i BSSEval.

Na końcu sekcji znajduje się abstrakcyjny interfejs `Metric` (`calculate`, `name`).

## 7. Steganaliza ([Steganalysis](../README.md#steganalysis))

Sekcja wyjaśnia, że metryki jakości i ataki nie odpowiadają na pytanie, czy *obecność* ukrytej wiadomości da się
wykryć — a to właśnie odróżnia steganografię od znakowania wodnego. Opisuje procedurę `measure_detectability`:
podział sygnałów na okna, parowanie okien nośnych z ich wersjami stego, rozłączny podział na zbiór uczący i testowy,
ekstrakcję cech (resztowe cechy Markowa i cechy log-widmowe) oraz klasyfikator zespołowy z dyskryminantami Fishera
na losowych podprzestrzeniach cech.

Interpretacja wyniku: dokładność bliska `0.5` oznacza zgadywanie (metoda niewykrywalna tymi cechami, próg `≤ 0.55`),
`1.0` — wykrywanie bezbłędne. Wynik jest dolnym ograniczeniem wykrywalności: brak wykrycia nie wyklucza skuteczności
silniejszych cech lub klasyfikatorów.

## 8. Model ataków i kanału ([Attack and channel model](../README.md#attacks))

Ataki modelują to, co spotyka sygnał stego między osadzeniem a odczytem: przetwarzanie, kompresję, transmisję,
odtworzenie i ponowne nagranie oraz celowe próby usunięcia wiadomości. Sekcja przedstawia zasady projektowe:

* **powtarzalność** — każdy atak losowy ma jawne ziarno;
* **jawne parametry** — etykiety siły (np. `mp3@strong`) są zamieniane na konkretne wartości zapisywane w wynikach;
* **zależność od częstotliwości próbkowania** — częstotliwości graniczne są wyliczane z `f_s` i sprawdzane względem
  częstotliwości Nyquista;
* **brak samonormalizacji** — atak nie kompensuje własnego efektu.

Tabela 6 grupuje ataki w rodziny: kodeki (MP3, AAC, Opus, Vorbis przez FFmpeg), szum, filtracja, zmiana
częstotliwości próbkowania, kwantyzacja, amplituda, przekształcenia czasowe, efekty akustyczne i złożone potoki
(np. rozmowa głosowa, transmisja przez powietrze). Sekcja opisuje też składnię specyfikacji ataków, gotowe zestawy
testowe (`quick`, `standard`, `full`) i odsyła do szczegółowej dokumentacji w [attacks.md](attacks.md).

## 9. Bibliografia ([References](../README.md#references))

* **9.1 Literatura** — 40 pozycji: publikacje źródłowe metod osadzania, metryk jakości, steganalizy oraz prace
  benchmarkowe. Numery `[n]` są używane w tabelach 1–5 i w sekcji 7.
* **9.2 Zasoby programistyczne** — projekty open source (oznaczone `[S1]`–`[S11]`), na których wzorowano się lub które
  są wykorzystywane przez poszczególne komponenty.

## 10. Licencja ([Licence](../README.md#licence))

Projekt jest wolnym oprogramowaniem udostępnianym na licencji GNU GPL w wersji 3.

## 11. Zależności zewnętrzne ([External dependencies](../README.md#dependencies))

Wymagania systemowe spoza Pythona: Microsoft Visual C++ Build Tools (do kompilacji PESQ) oraz FFmpeg (ataki kodekowe
i część konwersji formatów).

## 12. Autorzy ([Authors](../README.md#authors))

Autorzy projektu i ich afiliacja: Wojskowa Akademia Techniczna, Wydział Elektroniki.
