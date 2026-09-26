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
  `CapacityError` (podklasa `ValueError`) przy przekroczeniu pojemności oraz stabilność na sygnałach syntetycznych.
  Silnik eksperymentów korzysta z dwóch z nich: dekoduje zawsze nową instancją, a `CapacityError` zapisuje jako
  przekroczenie pojemności, a nie jako awarię. Sekcja wyjaśnia też, dlaczego siły osadzenia są definiowane względnie
  (np. względem normy ramki), a nie jako stałe bezwzględne.

## 2. Instalacja ([Installation](../README.md#install))

Instalacja pakietu z PyPI (`pip install the-a-files`) oraz tabela opcjonalnych rozszerzeń:

* `neural` — wytrenowane sieci do znakowania wodnego (AudioSeal, WavMark; PyTorch);
* `ai` — metoda FGAS i metryka MOSNet (TensorFlow);
* `experiments` — silnik eksperymentów;
* `platform` — platforma badawcza: REST API z bazą PostgreSQL i biblioteką korpusów;
* `dev` — narzędzia do testów i budowania pakietu.

## 3. Użycie ([Usage](../README.md#usage))

Uruchamianie wbudowanego przepływu ewaluacji poleceniem `taf-eval` (scenariusze `direct-no-metrics` i `full` lub własny
plik YAML) oraz przykład bezpośredniego użycia metody przez fabrykę `SteganographyMethodFactory`: osadzenie
i odczyt wiadomości.

## 4. Silnik eksperymentów, REST API i platforma webowa ([Experiment engine, REST API and web platform](../README.md#platform))

* **4.1 Silnik eksperymentów** — deklaratywna definicja eksperymentu (`ExperimentConfig`): zbiór danych, metody,
  metryki, ataki, długości wiadomości, liczba powtórzeń i ziarno losowości. Tabela przypisuje dziewięć projektów
  eksperymentów do czterech cech systemu ukrywania informacji:
  * niepostrzegalność — `perceptual_quality`;
  * odporność — `attack_robustness` oraz `robustness_curve` (krzywa dawka–odpowiedź BER względem siły jednego ataku,
    z punktem załamania);
  * pojemność — `embedding_capacity`;
  * bezpieczeństwo — `detectability` (steganaliza);
  * wielokryterialne — `tradeoff_curve` (krzywa kompromisu jakość–BER względem parametru metody, z frontem Pareto),
    `method_comparison` i `dataset_benchmark`;
  * eksploracyjny — `research_experiment`.

  Wszystkie projekty poza `detectability` wykonują pełny układ czynnikowy (pliki × metody × długości wiadomości ×
  powtórzenia × warianty ataków) i różnią się ustalonymi czynnikami oraz analizą. Metodę podaje się nazwą z katalogu
  albo specyfikacją z parametrami konstruktora (np. `"QIM_METHOD:step_scale=0.1"`). Obie krzywe definiuje
  `ParameterSweep` (cel, parametr, wartości od łagodnych do ostrych). Sekcja ma cztery podsekcje:
  * **Experimental protocol** — zasady prowadzenia prób:
    * jedno ziarno eksperymentu, zapisywane nawet wtedy, gdy zostało wylosowane;
    * wiadomości różnych długości są niezależne;
    * ziarno ataku zależy od pliku, powtórzenia i ataku, ale nie od metody, więc każda metoda trafia na ten sam szum
      (*common random numbers*);
    * dekodowanie odbywa się nową instancją metody;
    * metryki osadzenia są oddzielone od metryk szkody wyrządzonej przez atak;
    * awaria to nie błąd bitowy: ma własną kategorię i nie ma BER.
  * **Statistical analysis** — jednostką replikacji jest plik:
    * przedziały ufności z bootstrapu klastrowego po plikach;
    * sparowane testy rangowe (Wilcoxon albo Friedman z korektą Holma, wielkość efektu, różnica krytyczna Nemenyiego);
    * front Pareto zamiast arbitralnego wyniku ważonego;
    * pojemność liczona per plik, w bitach i w b/s.
  * **Evaluation block** — wspólny blok ewaluacji każdego przebiegu czynnikowego: rozkład BER dla każdej metody (średnia
    z przedziałem, mediana, SD, zakres, kwartyle, IQR) bez ataku i pod atakiem, odsetki odzysku (BER = 0, ≤ 1%, ≤ 5%),
    wzrost BER wywołany każdym atakiem (każdy plik porównany sam ze sobą), krzywe degradacji z testem trendu, jakość
    nośnik–stego oddzielona od szkody wyrządzonej przez atak, długość wiadomości, czas przetwarzania i współczynnik czasu
    rzeczywistego, korelacje Spearmana z testem permutacyjnym wewnątrz plików i korektą Holma oraz stwierdzenia
    wyliczone z tych liczb. Ustawienia (poziom ufności, liczba prób bootstrap, ziarna, testy, korekta) są zapisane
    razem z blokiem.
  * **Provenance** — manifest przebiegu: wersje pakietów i FFmpeg, commit źródeł, skróty SHA-256 plików wejściowych,
    rozwinięte ziarno oraz informacja, czy pomiary czasu są porównywalne.
  * **Reproduction of single trials and reports** — każdą zapisaną próbę da się dokładnie odtworzyć z ziarna (sygnał
    oryginalny, stego, po ataku i różnicowy). Raport z przebiegu powstaje w LaTeX-u (booktabs) lub w Markdownie, razem
    z akapitem *Experimental setup*.
* **4.2 Platforma badawcza** — eksperymenty (wersjonowane protokoły z pytaniem badawczym i hipotezą), przebiegi,
  wiersze wyników i zbiory danych są przechowywane w PostgreSQL. Bazę uruchamia `docker compose up -d db`, a API
  polecenie `taf-api`. Migracje są stosowane przy starcie, a konfigurację ustawiają zmienne `TAF_DATABASE_URL`,
  `TAF_DATA_DIR` i `TAF_MAX_CONCURRENT_RUNS`. Sekcja zawiera tabelę punktów końcowych (katalog, eksperymenty, przebiegi,
  eksporty i raporty, inspektor prób, postęp przez Server-Sent Events, zbiory danych). Klient webowy w `web/`
  (Next.js; angielski i polski; motyw jasny i ciemny) jest cienkim klientem API i nie należy do dystrybucji PyPI.
  Podsekcja **Evaluation corpora** opisuje katalog standardowych korpusów: mowę, mowę syntetyczną, muzykę i dźwięki
  otoczenia, z licencją, cytowaniem i DOI. Z otwartych korpusów platforma przygotowuje odtwarzalne podzbiory według
  zapisanej reguły: losowanie z ziarnem, równoważenie mówców, mono, resampling, FLAC i manifest SHA-256.
* **4.3 Rozszerzenia przez wtyczki** — metody, metryki i ataki z innych pakietów rejestrowane przez *entry points*
  (`taf.methods`, `taf.metrics`, `taf.attacks`) bez modyfikowania tego pakietu. Nazwy wbudowane mają pierwszeństwo.
  Automatyczny test pilnuje warstw: elementy składowe nie importują silnika, a silnik nie importuje warstwy HTTP.

## 5. Metody steganografii i znakowania wodnego ([Steganography and watermarking methods](../README.md#steganography-algorithms))

Tabela 1 zawiera 27 zaimplementowanych metod wraz z odwołaniami do publikacji źródłowych. Obejmują one metody
w dziedzinie czasu (LSB, echo, histogram, modyfikacja amplitudy niskich częstotliwości), w dziedzinach transformat
(DCT, DWT, LWT, SVD), rozpraszanie widma (DSSS, Improved Spread Spectrum), modulację indeksem kwantyzacji (QIM),
kodowanie fazy, metody adaptacyjne (AAC + kody STC) oraz metody neuronowe (FGAS, AudioSeal, WavMark).

Sekcja pokazuje też abstrakcyjny interfejs `SteganographyMethod` (`encode`, `decode`, `type`). Przypomina, że metody
wbudowane rejestruje się w fabryce i w typie `MethodType`, a zewnętrzne jako wtyczki (sekcja 4.3). Każda musi spełniać
kontrakt z sekcji 1.

## 6. Obiektywne metryki jakości ([Objective quality metrics](../README.md#metrics))

21 metryk porównujących sygnał nośny z sygnałem przetworzonym, pogrupowanych według mierzonej własności
(tabele 2–5, numeracja ciągła):

* **6.1 Metryki oparte na uczeniu maszynowym** — MOSNet, przewidujący średnią ocenę subiektywną (MOS);
* **6.2 Pogłos mowy** — BSD i SRMR;
* **6.3 Zrozumiałość mowy** — CSII, NCM, STOI;
* **6.4 Jakość mowy** — miary wierności sygnału i jakości percepcyjnej: SNR, SNRseg, fwSNRseg, LLR, WSS, odległości
  cepstralne (CD, MCD), PESQ, metryki złożone (Csig, Cbak, Covl), wSTMI, STGI, SI-SDR i BSSEval.

Wstęp sekcji wyjaśnia, że każda metryka deklaruje swój kierunek (`higher_is_better`), zamiast zgadywać go z nazwy.
Metryka zwracająca kilka liczb deklaruje nazwy składowych (`components`), a każda składowa jest raportowana osobno,
nie uśredniana. Przykłady:

* BSSEval: SDR, ISR, SIR, SAR oraz indeks permutacji, który nie jest rangowany;
* PESQ: surowy wynik P.862 i MOS-LQO;
* CSII: trzy indeksy;
* SRMR i MOSNet: wynik sygnału nośnego jako odniesienie.

STGI i wSTMI przy częstotliwości innej niż 10 kHz wykonują resampling.

Na końcu sekcji znajduje się abstrakcyjny interfejs `Metric` (`calculate`, `name`).

## 7. Steganaliza ([Steganalysis](../README.md#steganalysis))

Sekcja wyjaśnia, że metryki jakości i ataki nie odpowiadają na pytanie, czy *obecność* ukrytej wiadomości da się
wykryć — a to właśnie odróżnia steganografię od znakowania wodnego. Opisuje procedurę `measure_detectability`:
podział sygnałów na okna, parowanie okien nośnych z ich wersjami stego, rozłączny podział na zbiór uczący i testowy,
ekstrakcję cech (resztowe cechy Markowa i cechy log-widmowe) oraz klasyfikator zespołowy z dyskryminantami Fishera
na losowych podprzestrzeniach cech.

Interpretacja wyniku: dokładność bliska `0.5` oznacza zgadywanie, a `1.0` wykrywanie bezbłędne. Wynik zawiera 95%
przedział Wilsona oraz jednostronny dokładny test dwumianowy względem zgadywania (`significantly_detectable`). Starszy
znacznik `undetectable` (dokładność ≤ 0.55) to heurystyka na estymacie punktowej i przy małym zbiorze testowym może
się z testem nie zgadzać. Wynik jest dolnym ograniczeniem wykrywalności: brak wykrycia nie wyklucza skuteczności
silniejszych cech lub klasyfikatorów. Typ eksperymentu `detectability` uruchamia tę analizę dla wybranych metod
i długości wiadomości.

## 8. Model ataków i kanału ([Attack and channel model](../README.md#attacks))

Ataki modelują to, co spotyka sygnał stego między osadzeniem a odczytem: przetwarzanie, kompresję, transmisję,
odtworzenie i ponowne nagranie oraz celowe próby usunięcia wiadomości. Sekcja przedstawia zasady projektowe:

* **powtarzalność** — każdy atak losowy ma jawne ziarno, a w eksperymencie ziarno jest wyprowadzane dla każdej próby
  (chyba że specyfikacja podaje własne);
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
