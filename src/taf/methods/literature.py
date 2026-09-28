"""Curated primary-source evidence, independent of executable method availability.

Numbers are paper-reported, not TAF replications. Each quantitative observation
retains its experimental condition and table/section locator. Missing values
mean unverified/not applicable, never zero. No cross-paper ranking is implied.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Observation:
    metric: str
    value: float
    unit: str
    condition: str
    locator: str


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str
    family: str
    purpose: str
    url: str
    doi: str | None
    arxiv: str | None
    datasets: tuple[str, ...]
    metrics: tuple[str, ...]
    attacks: tuple[str, ...]
    payload: str
    results: tuple[Observation, ...]
    limitations: str
    source_url: str
    implemented_methods: tuple[str, ...] = ()
    verified_on: str = "2026-09-26"
    evidence: str = "primary paper"


PAPERS = (
    Paper(
        id="hide-and-speak", title="Hide and Speak: Towards Deep Neural Networks for Speech Steganography",
        authors=("Felix Kreuk", "Yossi Adi", "Bhiksha Raj", "Rita Singh", "Joseph Keshet"),
        year=2020, venue="Interspeech 2020 (preprint 2019)", family="neural", purpose="steganography",
        url="https://www.isca-archive.org/interspeech_2020/kreuk20b_interspeech.html", doi="10.21437/Interspeech.2020-2380", arxiv="1902.03083",
        datasets=("TIMIT", "YOHO"), metrics=("carrier SNR", "message SNR", "absolute error", "ABX", "WER", "CER"),
        attacks=("MP3", "additive white Gaussian noise", "sample-rate reduction"),
        payload="1, 3 or 5 speech messages per carrier; waveform recovery, not a bit-exact binary channel.",
        results=(Observation("carrier SNR", 28.27, "dB", "TIMIT, single message, Ours without adversarial loss", "Table 1"),
                 Observation("message SNR", 8.76, "dB", "TIMIT, single message, Ours without adversarial loss", "Table 1"),
                 Observation("ABX correct discrimination", 51.2, "%", "50 clips, 20 judgments each, combined TIMIT/YOHO", "Section 5.1")),
        limitations="Audio reconstruction scores cannot be compared directly with BER or binary bps. Listening results do not establish resistance to machine steganalysis.",
        source_url="https://arxiv.org/pdf/1902.03083v2",
    ),
    Paper(
        id="dear", title="DeAR: A Deep-learning-based Audio Re-recording Resilient Watermarking",
        authors=("Chang Liu", "Jie Zhang", "Han Fang", "Zehua Ma", "Weiming Zhang", "Nenghai Yu"),
        year=2023, venue="AAAI 2023 (preprint 2022)", family="neural", purpose="watermarking",
        url="https://ojs.aaai.org/index.php/AAAI/article/view/26550", doi="10.1609/aaai.v37i11.26550", arxiv="2212.02339",
        datasets=("FMA",), metrics=("SNR", "bit recovery accuracy"),
        attacks=("re-recording", "reverberation", "band-pass filtering", "Gaussian noise"),
        payload="100 bits per 500,000 samples at 44.1 kHz in the main test; 64/169/225-bit ablations.",
        results=(Observation("SNR", 25.86, "dB", "Main 100-bit FMA experiment", "Table 1"),
                 Observation("bit recovery accuracy", 98.55, "%", "Default DeAR, physical re-recording at 20 cm", "Table 4")),
        limitations="200 test clips. Synchronization searches shifts using the highest recovery accuracy; this is not equivalent to TAF's blind extraction protocol.",
        source_url="https://arxiv.org/pdf/2212.02339v4",
    ),
    Paper(
        id="silentcipher", title="SilentCipher: Deep Audio Watermarking",
        authors=("Mayank Kumar Singh", "Naoya Takahashi", "Weihsiang Liao", "Yuki Mitsufuji"),
        year=2024, venue="Interspeech 2024", family="neural", purpose="watermarking",
        url="https://www.isca-archive.org/interspeech_2024/singh24_interspeech.html",
        doi="10.21437/Interspeech.2024-174", arxiv="2406.03822",
        datasets=("VCTK", "NUS-48E", "NHSS", "internal audio collections"),
        metrics=("message accuracy", "SDR", "encoding runtime", "subjective inaudibility"),
        attacks=("Gaussian noise", "50% cropping", "equalization", "mixing", "quantization", "time jitter", "resampling", "MP3", "OGG", "AAC"),
        payload="SC-16 trains with repeated 32-bit messages; SC-44 with repeated 40-bit messages. Repetition is not new information capacity.",
        results=(Observation("mean message accuracy", 98.93, "%", "SC-16, 6-second clips, Table 1 conditions", "Table 1"),
                 Observation("mean message accuracy", 99.96, "%", "SC-44, 24-second clips, Table 1 conditions", "Table 1")),
        limitations="Different rates/durations and partly private data; reported accuracy is not a directly comparable TAF BER estimate.",
        source_url="https://www.isca-archive.org/interspeech_2024/singh24_interspeech.pdf",
    ),
    Paper(
        id="ideaw", title="IDEAW: Robust Neural Audio Watermarking with Invertible Dual-Embedding",
        authors=("Pengcheng Li", "Xulong Zhang", "Jing Xiao", "Jianzong Wang"),
        year=2024, venue="EMNLP 2024", family="neural", purpose="watermarking",
        url="https://aclanthology.org/2024.emnlp-main.258/", doi="10.18653/v1/2024.emnlp-main.258", arxiv="2409.19627",
        datasets=("VCTK", "FMA"), metrics=("SNR", "bit accuracy", "capacity", "locating efficiency"),
        attacks=("Gaussian noise", "low-pass", "MP3", "quantization", "resampling", "dropout", "amplitude modification", "time stretch"),
        payload="20/32/56 gross bps include a 10-bit locating code; IDEAW46+10 carries 46 message bits per one-second segment at 16 kHz.",
        results=(Observation("SNR", 35.41, "dB", "IDEAW46+10, Table 1 evaluation", "Table 1"),
                 Observation("bit accuracy", 99.44, "%", "IDEAW46+10, locating code and message combined", "Table 1")),
        limitations="Gross rate and accuracy include synchronization bits. Do not equate 56 gross bps with 56 useful message bps.",
        source_url="https://aclanthology.org/2024.emnlp-main.258.pdf",
    ),
)


def literature_entries():
    return [asdict(paper) for paper in PAPERS]
