# Polski przewodnik

The A-Files s?u?y do powtarzalnego por?wnywania metod steganografii i znakowania wodnego w audio.
Ocenia odzyskanie wiadomo?ci, zniekszta?cenia sygna?u, odporno?? na przetwarzanie, pojemno?? i wykrywalno??.

## Dokumentacja

- [Instalacja](installation.md) ? Python, opcjonalne modele i pierwsze uruchomienie.
- [Interfejs badawczy](ui.md) ? uruchomienie UI, przygotowanie protoko?u, analiza i eksport wynik?w.
- [27 metod](methods.md) ? mechanizmy osadzania, identyfikatory, ograniczenia implementacji i publikacje.
- [Ataki i kana?y](attacks.md) ? opis ka?dej transformacji, parametr?w i scenariuszy z?o?onych.
- [25 metryk](metrics.md) ? znaczenie wyniku, kierunek interpretacji i ?r?d?a naukowe.
- [Protok??](protocol.md) i [statystyka](experiments.md) ? zasady por?wnywania metod i szacowania niepewno?ci.
- [Wiadomo?ci i miary](research-capabilities.md) ? formaty danych, definicje miar i odtwarzanie eksperyment?w.
- [Bibliografia](references.md) ? publikacje stanowi?ce podstaw? opis?w.

## Praca w UI

Uruchom baz? PostgreSQL, API i klienta web wed?ug [instrukcji](ui.md#start-locally), nast?pnie otw?rz
**http://localhost:3000**. J?zyk polski mo?na wybra? w nag??wku aplikacji.

1. Wybierz lub przygotuj zbi?r nagra? w **Datasets**.
2. Utw?rz eksperyment, wybierz projekt badania, metody, wiadomo?ci, ataki i metryki.
3. Sprawd? plan wykonania i ostrze?enia, zapisz protok?? i uruchom badanie.
4. Por?wnaj wyniki i b??dy, obejrzyj statystyki oraz ods?uchaj sygna?y w inspektorze pr?b.
5. Pobierz tabele CSV, konfiguracj?, manifest i raport Markdown lub LaTeX.

Wynik metryki jako?ci nie oznacza odporno?ci ani niewykrywalno?ci wiadomo?ci. Opisy rozr??niaj?
algorytm z publikacji od jego lokalnej adaptacji; wyniki z artyku??w nie s? wynikami pomiar?w wykonanych w TAF.
GitHub Pages udost?pnia dokumentacj?. Aplikacja badawcza wymaga dzia?aj?cego API i bazy danych.
