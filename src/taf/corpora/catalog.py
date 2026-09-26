"""Standard corpora for evaluating audio steganography and watermarking.

Results are only comparable when they are obtained on material others can
obtain too. This catalogue lists the corpora the literature evaluates on,
with licence and citation, grouped by domain:

* **speech** - read and studio speech in English, the dominant test material
  for speech steganography (LibriSpeech, VCTK, TIMIT, TSP) and for the
  watermarking of generated speech (LJSpeech, LibriTTS-R, EARS);
* **music** - full-band stereo music, the classical watermarking material
  (MUSDB18-HQ, GTZAN, FMA);
* **environmental** - non-speech everyday sounds (ESC-50);
* **synthetic_speech** - speech produced by speech synthesis and voice
  conversion, the target of current generative-audio watermarking
  (ASVspoof 5, MLAAD).

Every download URL was checked when the entry was added. Corpora that
require registration or a licence (TIMIT, Common Voice, ASVspoof 5) or whose
hosting is unreliable are listed with ``download=None``: they are registered
from a local copy instead. Nothing is downloaded until asked for, and a
prepared subset records the archive digest and the selection rule, so the
exact material of an experiment can be rebuilt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DOMAINS = ("speech", "synthetic_speech", "music", "environmental", "synthetic_signal")


@dataclass(frozen=True)
class Download:
    url: str
    archive: str  # "tar.gz", "tar.bz2" or "zip"
    size_mb: float | None = None


@dataclass(frozen=True)
class Corpus:
    id: str
    name: str
    domain: str
    language: str | None
    description: str
    license: str
    citation: str
    year: int
    native_sample_rate: int | None
    doi: str | None = None
    url: str | None = None
    download: Download | None = None
    #: Glob, relative to the archive root, of the audio files to consider.
    audio_glob: str = "**/*.wav"
    #: Path component (counted from the end, 1 = parent directory) naming the
    #: speaker, for speaker-balanced selection; ``None`` when not applicable.
    speaker_level: int | None = None
    #: Substring every selected path must contain (e.g. one microphone).
    path_filter: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, Any]:
        data = asdict(self)
        data["downloadable"] = self.download is not None
        return data


CORPORA: tuple[Corpus, ...] = (
    Corpus(
        id="librispeech_test_clean",
        name="LibriSpeech test-clean",
        domain="speech",
        language="en",
        description="Read English audiobook speech, 40 speakers, the clean test partition.",
        license="CC BY 4.0",
        citation="Panayotov, Chen, Povey, Khudanpur, 'LibriSpeech: an ASR corpus based on public domain audio books', ICASSP 2015",
        year=2015,
        native_sample_rate=16000,
        doi="10.1109/ICASSP.2015.7178964",
        url="https://www.openslr.org/12",
        download=Download("https://www.openslr.org/resources/12/test-clean.tar.gz", "tar.gz", 347),
        audio_glob="**/*.flac",
        speaker_level=2,
        tags=("standard", "speech-watermarking"),
    ),
    Corpus(
        id="librispeech_dev_clean",
        name="LibriSpeech dev-clean",
        domain="speech",
        language="en",
        description="Read English audiobook speech, 40 speakers, the clean development partition.",
        license="CC BY 4.0",
        citation="Panayotov, Chen, Povey, Khudanpur, 'LibriSpeech: an ASR corpus based on public domain audio books', ICASSP 2015",
        year=2015,
        native_sample_rate=16000,
        doi="10.1109/ICASSP.2015.7178964",
        url="https://www.openslr.org/12",
        download=Download("https://www.openslr.org/resources/12/dev-clean.tar.gz", "tar.gz", 338),
        audio_glob="**/*.flac",
        speaker_level=2,
        tags=("standard",),
    ),
    Corpus(
        id="mini_librispeech",
        name="Mini LibriSpeech dev-clean-2",
        domain="speech",
        language="en",
        description="Small LibriSpeech subset for quick experiments and continuous integration.",
        license="CC BY 4.0",
        citation="Panayotov, Chen, Povey, Khudanpur, 'LibriSpeech: an ASR corpus based on public domain audio books', ICASSP 2015",
        year=2015,
        native_sample_rate=16000,
        doi="10.1109/ICASSP.2015.7178964",
        url="https://www.openslr.org/31",
        download=Download("https://www.openslr.org/resources/31/dev-clean-2.tar.gz", "tar.gz", 126),
        audio_glob="**/*.flac",
        speaker_level=2,
        tags=("quick",),
    ),
    Corpus(
        id="vctk",
        name="CSTR VCTK 0.92",
        domain="speech",
        language="en",
        description="110 English speakers with various accents, studio recordings (microphone 1).",
        license="CC BY 4.0",
        citation="Yamagishi, Veaux, MacDonald, 'CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)', University of Edinburgh, 2019",
        year=2019,
        native_sample_rate=48000,
        doi="10.7488/ds/2645",
        url="https://datashare.ed.ac.uk/handle/10283/3443",
        download=Download("https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip", "zip", 11700),
        audio_glob="**/*.flac",
        speaker_level=1,
        path_filter="_mic1",
        tags=("standard", "multi-speaker"),
    ),
    Corpus(
        id="tsp",
        name="TSP speech database (48 kHz)",
        domain="speech",
        language="en",
        description="Phonetically balanced sentences by 24 speakers, the classic material of speech quality measurement.",
        license="Free for research use with citation",
        citation="Kabal, 'TSP Speech Database', McGill University, Database Version 1.0, 2002",
        year=2002,
        native_sample_rate=48000,
        url="https://www.mmsp.ece.mcgill.ca/Documents/Data/",
        download=Download("https://www.mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip", "zip", 254),
        audio_glob="**/*.wav",
        speaker_level=1,
        tags=("standard", "speech-quality"),
    ),
    Corpus(
        id="ljspeech",
        name="LJSpeech 1.1",
        domain="speech",
        language="en",
        description="13,100 clips of a single speaker; the reference corpus of neural speech synthesis and of generated-speech watermarking.",
        license="Public domain",
        citation="Ito, Johnson, 'The LJ Speech Dataset', 2017",
        year=2017,
        native_sample_rate=22050,
        url="https://keithito.com/LJ-Speech-Dataset/",
        download=Download("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", "tar.bz2", 2600),
        audio_glob="**/*.wav",
        tags=("generative-audio",),
    ),
    Corpus(
        id="libritts_r_test_clean",
        name="LibriTTS-R test-clean",
        domain="speech",
        language="en",
        description="Speech-restored LibriTTS: studio-like quality multi-speaker speech used to train current TTS models.",
        license="CC BY 4.0",
        citation="Koizumi et al., 'LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus', Interspeech 2023",
        year=2023,
        native_sample_rate=24000,
        doi="10.21437/Interspeech.2023-1584",
        url="https://www.openslr.org/141",
        download=Download("https://www.openslr.org/resources/141/test_clean.tar.gz", "tar.gz", 1295),
        audio_glob="**/*.wav",
        speaker_level=2,
        tags=("generative-audio", "recent"),
    ),
    Corpus(
        id="ears_p001",
        name="EARS (speaker p001)",
        domain="speech",
        language="en",
        description="Anechoic full-band 48 kHz speech with reading, emotions and non-verbal sounds; one speaker's recordings.",
        license="CC BY-NC 4.0",
        citation="Richter et al., 'EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation', Interspeech 2024",
        year=2024,
        native_sample_rate=48000,
        doi="10.21437/Interspeech.2024-153",
        url="https://github.com/facebookresearch/ears_dataset",
        download=Download("https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p001.zip", "zip", 592),
        audio_glob="**/*.wav",
        tags=("full-band", "recent"),
    ),
    Corpus(
        id="esc50",
        name="ESC-50",
        domain="environmental",
        language=None,
        description="2,000 five-second environmental recordings in 50 classes.",
        license="CC BY-NC 3.0",
        citation="Piczak, 'ESC: Dataset for Environmental Sound Classification', ACM Multimedia 2015",
        year=2015,
        native_sample_rate=44100,
        doi="10.1145/2733373.2806390",
        url="https://github.com/karolpiczak/ESC-50",
        download=Download("https://github.com/karoldvl/ESC-50/archive/master.zip", "zip", 600),
        audio_glob="**/audio/*.wav",
        tags=("non-speech",),
    ),
    Corpus(
        id="musdb18hq",
        name="MUSDB18-HQ",
        domain="music",
        language=None,
        description="150 full-length uncompressed stereo music tracks; the mixtures are used as covers.",
        license="Research use (see Zenodo record)",
        citation="Rafii, Liutkus, Stöter, Mimilakis, Bittner, 'MUSDB18-HQ - an uncompressed version of MUSDB18', 2019",
        year=2019,
        native_sample_rate=44100,
        doi="10.5281/zenodo.3338373",
        url="https://zenodo.org/records/3338373",
        download=Download("https://zenodo.org/records/3338373/files/musdb18hq.zip", "zip", 22657),
        audio_glob="**/mixture.wav",
        tags=("music-watermarking", "large"),
    ),
    Corpus(
        id="timit",
        name="TIMIT",
        domain="speech",
        language="en",
        description="630 speakers of eight dialects of American English; distributed by the LDC under licence.",
        license="LDC licence (LDC93S1)",
        citation="Garofolo et al., 'TIMIT Acoustic-Phonetic Continuous Speech Corpus', LDC93S1, 1993",
        year=1993,
        native_sample_rate=16000,
        doi="10.35111/17gk-bn40",
        url="https://catalog.ldc.upenn.edu/LDC93S1",
        audio_glob="**/*.wav",
        speaker_level=1,
        tags=("standard", "licensed"),
    ),
    Corpus(
        id="common_voice",
        name="Mozilla Common Voice",
        domain="speech",
        language="multi",
        description="Crowd-sourced read speech in over 100 languages; downloaded per language after sign-in.",
        license="CC0",
        citation="Ardila et al., 'Common Voice: A Massively-Multilingual Speech Corpus', LREC 2020",
        year=2020,
        native_sample_rate=48000,
        url="https://commonvoice.mozilla.org/datasets",
        audio_glob="**/*.mp3",
        tags=("multilingual",),
    ),
    Corpus(
        id="asvspoof5",
        name="ASVspoof 5",
        domain="synthetic_speech",
        language="en",
        description="Bona fide and spoofed (TTS and voice conversion) speech; distributed after registration.",
        license="Registration required",
        citation="Wang et al., 'ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale', ASVspoof Workshop 2024",
        year=2024,
        native_sample_rate=16000,
        url="https://www.asvspoof.org/",
        audio_glob="**/*.flac",
        tags=("generative-audio", "recent", "licensed"),
    ),
    Corpus(
        id="mlaad",
        name="MLAAD",
        domain="synthetic_speech",
        language="multi",
        description="Multi-language audio anti-spoofing dataset of speech generated by dozens of TTS systems.",
        license="CC BY-NC 4.0",
        citation="Müller et al., 'MLAAD: The Multi-Language Audio Anti-Spoofing Dataset', IJCNN 2024",
        year=2024,
        native_sample_rate=None,
        url="https://huggingface.co/datasets/mueller91/MLAAD",
        audio_glob="**/*.wav",
        tags=("generative-audio", "recent"),
    ),
    Corpus(
        id="gtzan",
        name="GTZAN",
        domain="music",
        language=None,
        description="1,000 thirty-second music excerpts in ten genres; the original host is no longer reliable.",
        license="Unclear (research use)",
        citation="Tzanetakis, Cook, 'Musical genre classification of audio signals', IEEE TSAP 2002",
        year=2002,
        native_sample_rate=22050,
        doi="10.1109/TSA.2002.800560",
        audio_glob="**/*.wav",
        tags=("music-watermarking",),
    ),
    Corpus(
        id="noizeus",
        name="NOIZEUS",
        domain="speech",
        language="en",
        description="IEEE sentences with and without real-world noise, the reference set of speech quality metrics.",
        license="Free for research use",
        citation="Hu, Loizou, 'Subjective comparison and evaluation of speech enhancement algorithms', Speech Communication 2007",
        year=2007,
        native_sample_rate=8000,
        doi="10.1016/j.specom.2006.12.006",
        url="https://ecs.utdallas.edu/loizou/speech/noizeus/",
        audio_glob="**/*.wav",
        tags=("speech-quality",),
    ),
)

CORPORA_BY_ID = {corpus.id: corpus for corpus in CORPORA}


def get_corpus(corpus_id: str) -> Corpus:
    try:
        return CORPORA_BY_ID[corpus_id]
    except KeyError as error:
        raise KeyError(f"Unknown corpus {corpus_id!r}; known: {sorted(CORPORA_BY_ID)}") from error


__all__ = ["CORPORA", "CORPORA_BY_ID", "Corpus", "DOMAINS", "Download", "get_corpus"]
