from __future__ import annotations

import sys
import traceback
from inspect import signature
from pathlib import Path

import numpy as np

_SOURCE_PATH = Path(__file__).resolve().parents[2]
_source_path_text = str(_SOURCE_PATH)
if _source_path_text in sys.path:
    sys.path.remove(_source_path_text)
sys.path.insert(0, _source_path_text)

from taf.ui.helpers import (
    MAX_MESSAGE_BITS,
    MIN_MESSAGE_BITS,
    apply_attacks,
    bits_to_string,
    calculate_bit_accuracy,
    calculate_correct_bits,
    create_metric,
    create_method,
    discover_attacks,
    discover_methods,
    discover_metrics,
    generate_message,
    load_uploaded_audio,
    quick_attack_options,
    quick_metric_options,
    quick_method_options,
    save_audio_to_wav,
    to_mono,
    validate_binary_message,
)


def main() -> None:
    import sys
    from streamlit.web import cli as streamlit_cli

    script_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(script_path), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


def _is_running_with_streamlit() -> bool:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    return get_script_run_ctx(suppress_warning=True) is not None


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="The A-Files", layout="wide")
    _apply_material_styles(st)
    st.title("The A-Files")

    page = _render_navigation(st)
    if page == "Dashboard":
        _render_dashboard(st)
    else:
        _render_quick_analysis(st)


def _apply_material_styles(st) -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            min-height: 44px;
            border-radius: 8px;
            font-weight: 500;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #1a73e8;
            border-color: #1a73e8;
            color: #ffffff;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
            border: 1px solid rgba(49, 51, 63, 0.10) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
            background: rgba(255,255,255,0.03);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(26, 115, 232, 0.25) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 8px 24px rgba(26,115,232,0.07);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] p {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_navigation(st) -> str:
    pages = (
        ("Dashboard", "Dashboard", ":material/dashboard:"),
        ("Quick analysis", "Quick analysis", ":material/fingerprint:"),
    )
    st.session_state.setdefault("selected_page", "Dashboard")

    def _select(page_key: str) -> None:
        st.session_state.selected_page = page_key

    st.sidebar.caption("Menu")
    for page_key, label, icon in pages:
        button_kwargs = _button_kwargs(st.sidebar.button, width="stretch", icon=icon)
        st.sidebar.button(
            label,
            key=f"nav_{page_key}",
            type="primary" if st.session_state.selected_page == page_key else "secondary",
            on_click=_select,
            args=(page_key,),
            **button_kwargs,
        )

    return st.session_state.selected_page


def _render_dashboard(st) -> None:
    st.header("Dashboard")

    methods, metrics, attacks = [], [], []
    errors: list[tuple[str, Exception]] = []

    for label, loader in (
        ("methods", discover_methods),
        ("metrics", discover_metrics),
        ("attacks", discover_attacks),
    ):
        try:
            value = loader()
        except Exception as exc:
            value = []
            errors.append((label, exc))

        if label == "methods":
            methods = value
        elif label == "metrics":
            metrics = value
        else:
            attacks = value

    col_methods, col_metrics, col_attacks = st.columns(3)
    col_methods.metric("Methods", len(methods))
    col_metrics.metric("Metrics", len(metrics))
    col_attacks.metric("Attacks", len(attacks))

    if errors:
        with st.expander("Discovery error details"):
            for label, exc in errors:
                st.error(f"Could not load {label}: {exc}")
                st.code("".join(traceback.format_exception(exc)))

    st.subheader("Steganography methods")
    _render_stretch_dataframe(st, methods)

    st.subheader("Metrics")
    _render_stretch_dataframe(st, metrics)

    st.subheader("Attacks")
    _render_stretch_dataframe(st, attacks)


def _render_stretch_dataframe(st, data) -> None:
    if "width" in signature(st.dataframe).parameters:
        st.dataframe(data, width="stretch", hide_index=True)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True)


def _button_kwargs(widget, *, width: str, icon: str | None = None) -> dict[str, object]:
    parameters = signature(widget).parameters
    kwargs: dict[str, object] = {}
    if "width" in parameters:
        kwargs["width"] = width
    elif width == "stretch" and "use_container_width" in parameters:
        kwargs["use_container_width"] = True
    if icon is not None and "icon" in parameters:
        kwargs["icon"] = icon
    return kwargs


def _render_step_header(st, step: int, total_steps: int, title: str, *, enabled: bool) -> None:
    state = "active" if enabled else "waiting"
    st.caption(f"Step {step}/{total_steps} - {state}")
    st.subheader(title)


def _render_quick_analysis(st) -> None:
    st.header("Quick analysis")

    total_steps = 7
    step_counter = st.empty()
    step_progress = st.progress(0.0)

    audio_samples = None
    sample_rate = 16000
    uploaded_file = None
    audio_loaded = False

    with st.container(border=True):
        _render_step_header(st, 1, total_steps, "Audio file", enabled=True)
        uploaded_file = st.file_uploader("Audio file", type=["wav", "flac"])
        if uploaded_file is not None:
            try:
                audio_samples, sample_rate = load_uploaded_audio(uploaded_file)
                audio_samples, converted_to_mono = to_mono(audio_samples)
                audio_loaded = True
            except Exception as exc:
                st.error(f"Could not load audio file: {exc}")
                _debug_exception(st, exc)
            else:
                st.caption(
                    f"{uploaded_file.name} | {sample_rate} Hz | {audio_samples.shape[0]} samples"
                )
                if converted_to_mono:
                    st.info(
                        "The multichannel file was converted to mono by averaging channels."
                    )

    method_options = quick_method_options(sample_rate)
    attack_options = quick_attack_options()
    metric_options = quick_metric_options()
    selected_option = None
    message_bits: list[int] = []
    message_valid = False

    with st.container(border=True):
        _render_step_header(st, 2, total_steps, "Method and message", enabled=audio_loaded)
        if not method_options:
            st.error("No methods are available for quick analysis.")
        else:
            selected_option = st.selectbox(
                "Steganography method",
                method_options,
                format_func=lambda option: f"{option.name} ({option.class_name})",
                disabled=not audio_loaded,
            )

        message_length = st.slider(
            "Message length (bits)",
            min_value=MIN_MESSAGE_BITS,
            max_value=MAX_MESSAGE_BITS,
            value=16,
            step=1,
            disabled=not audio_loaded,
        )

        _ensure_generated_message(st, message_length)
        regenerate_kwargs = _button_kwargs(st.button, width="stretch", icon=":material/refresh:")
        if st.button("Generate again", disabled=not audio_loaded, **regenerate_kwargs):
            st.session_state.generated_message_bits = generate_message(message_length)
            st.session_state.generated_message_length = message_length

        generated_bits = st.session_state.generated_message_bits
        st.text_input(
            "Generated message",
            value=bits_to_string(generated_bits),
            disabled=True,
        )

        manual_message = st.text_input(
            "Custom binary message (optional)",
            disabled=not audio_loaded,
        )
        message_bits = generated_bits
        message_valid = audio_loaded and selected_option is not None
        if audio_loaded and manual_message.strip():
            try:
                message_bits = validate_binary_message(manual_message)
            except ValueError as exc:
                message_valid = False
                st.error(str(exc))

    context_key = None
    if uploaded_file is not None and selected_option is not None and message_valid:
        context_key = _quick_context_key(uploaded_file, sample_rate, selected_option, message_bits)

    encoded_ready = (
        context_key is not None
        and st.session_state.get("encoded_context_key") == context_key
        and "encoded_samples" in st.session_state
    )

    with st.container(border=True):
        _render_step_header(st, 3, total_steps, "Encoding", enabled=message_valid)
        encode_kwargs = _button_kwargs(st.button, width="stretch", icon=":material/lock:")
        if st.button("Encode", type="primary", disabled=not message_valid, **encode_kwargs):
            try:
                with st.spinner("Encoding message..."):
                    method = create_method(sample_rate, selected_option.method_type)
                    encoded_samples = method.encode(audio_samples.copy(), list(message_bits))
                    encoded_samples = np.asarray(encoded_samples)
                    output_bytes = save_audio_to_wav(encoded_samples, sample_rate)
            except Exception as exc:
                st.error(f"Could not encode message: {exc}")
                _debug_exception(st, exc)
            else:
                original_stem = Path(uploaded_file.name).stem
                st.session_state.encoded_samples = encoded_samples
                st.session_state.encoded_audio_bytes = output_bytes
                st.session_state.encoded_file_name = f"{original_stem}_encoded.wav"
                st.session_state.original_file_name = uploaded_file.name
                st.session_state.selected_method = selected_option.name
                st.session_state.original_message_bits = list(message_bits)
                st.session_state.sample_rate = sample_rate
                st.session_state.encoded_context_key = context_key
                st.session_state.pop("decoded_context_key", None)
                st.session_state.pop("decoded_bits", None)
                st.session_state.pop("attack_context_key", None)
                st.session_state.pop("attacked_samples", None)
                st.session_state.pop("attacked_sample_rate", None)
                st.session_state.pop("selected_attack_names", None)
                st.session_state.pop("metrics_context_key", None)
                st.session_state.pop("metric_rows", None)

                st.success("Message encoded.")
                encoded_ready = True

    with st.container(border=True):
        _render_step_header(st, 4, total_steps, "Download", enabled=encoded_ready)
        if encoded_ready:
            _render_encoded_summary(st, audio_samples)
        st.download_button(
            label="Download encoded file",
            data=st.session_state.get("encoded_audio_bytes", b""),
            file_name=st.session_state.get("encoded_file_name", "encoded.wav"),
            mime="audio/wav",
            disabled=not encoded_ready,
            **_button_kwargs(st.download_button, width="stretch", icon=":material/download:"),
        )

    selected_attack_options = []
    attack_context_key = None
    attack_ready = encoded_ready
    processed_samples = st.session_state.get("encoded_samples")
    processed_sample_rate = sample_rate
    selected_attack_names: list[str] = []

    with st.container(border=True):
        _render_step_header(st, 5, total_steps, "Attacks", enabled=encoded_ready)
        selected_attack_options = st.multiselect(
            "Attacks to apply",
            attack_options,
            format_func=lambda option: option.name,
            disabled=not encoded_ready,
        )
        selected_attack_names = [option.name for option in selected_attack_options]
        st.caption(f"Selected attacks: {len(selected_attack_options)}")

        if context_key is not None:
            attack_context_key = (
                context_key,
                tuple(option.name for option in selected_attack_options),
            )

        attack_ready = (
            encoded_ready
            and not selected_attack_options
        ) or (
            attack_context_key is not None
            and st.session_state.get("attack_context_key") == attack_context_key
            and "attacked_samples" in st.session_state
        )

        apply_attack_kwargs = _button_kwargs(st.button, width="stretch", icon=":material/waves:")
        if st.button(
            "Apply attacks",
            type="primary",
            disabled=not encoded_ready or not selected_attack_options,
            **apply_attack_kwargs,
        ):
            try:
                with st.spinner("Applying attacks..."):
                    attacked_samples, attacked_sample_rate = apply_attacks(
                        st.session_state.encoded_samples,
                        sample_rate,
                        selected_attack_options,
                    )
            except Exception as exc:
                st.error(f"Could not apply attacks: {exc}")
                _debug_exception(st, exc)
            else:
                st.session_state.attacked_samples = attacked_samples
                st.session_state.attacked_sample_rate = attacked_sample_rate
                st.session_state.attack_context_key = attack_context_key
                st.session_state.selected_attack_names = [
                    option.name for option in selected_attack_options
                ]
                st.session_state.pop("decoded_context_key", None)
                st.session_state.pop("decoded_bits", None)
                st.session_state.pop("metrics_context_key", None)
                st.session_state.pop("metric_rows", None)
                st.success("Attacks applied.")
                attack_ready = True

        if not selected_attack_options:
            st.info("No attacks selected. Decoding will use the encoded signal.")
        elif attack_ready:
            st.write("Applied attacks")
            st.code(", ".join(st.session_state.selected_attack_names))

        if attack_ready and selected_attack_options:
            processed_samples = st.session_state.attacked_samples
            processed_sample_rate = st.session_state.attacked_sample_rate

    decoded_ready = (
        attack_ready
        and st.session_state.get("decoded_context_key") == attack_context_key
        and "decoded_bits" in st.session_state
    )

    with st.container(border=True):
        _render_step_header(st, 6, total_steps, "Decoding", enabled=attack_ready)
        decode_kwargs = _button_kwargs(st.button, width="stretch", icon=":material/key:")
        if st.button("Decode", type="primary", disabled=not attack_ready, **decode_kwargs):
            try:
                with st.spinner("Decoding message..."):
                    method = create_method(sample_rate, selected_option.method_type)
                    original_bits = st.session_state.original_message_bits
                    decoded_bits = method.decode(processed_samples, len(original_bits))
                    decoded_bits = [int(bit) for bit in decoded_bits]
            except Exception as exc:
                st.error(f"Could not decode message: {exc}")
                _debug_exception(st, exc)
            else:
                st.session_state.decoded_bits = decoded_bits
                st.session_state.decoded_context_key = attack_context_key
                decoded_ready = True

        if decoded_ready:
            _render_decode_result(st, st.session_state.original_message_bits, st.session_state.decoded_bits)

    with st.container(border=True):
        _render_step_header(st, 7, total_steps, "Metrics", enabled=attack_ready)
        selected_metric_options = st.multiselect(
            "Metrics to calculate",
            metric_options,
            format_func=lambda option: f"{option.name} ({option.category})",
            disabled=not attack_ready,
        )
        st.caption(f"Selected metrics: {len(selected_metric_options)}")

        metrics_context_key = None
        if attack_context_key is not None:
            metrics_context_key = (
                attack_context_key,
                tuple(option.metric_type.name for option in selected_metric_options),
            )

        metrics_ready = (
            metrics_context_key is not None
            and st.session_state.get("metrics_context_key") == metrics_context_key
            and "metric_rows" in st.session_state
        )

        calculate_metrics_kwargs = _button_kwargs(
            st.button,
            width="stretch",
            icon=":material/analytics:",
        )
        if st.button(
            "Calculate metrics",
            type="primary",
            disabled=not attack_ready or not selected_metric_options,
            **calculate_metrics_kwargs,
        ):
            with st.spinner("Calculating metrics..."):
                metric_rows = _calculate_selected_metrics(
                    selected_metric_options,
                    audio_samples,
                    processed_samples,
                    processed_sample_rate,
                )
            st.session_state.metric_rows = metric_rows
            st.session_state.metrics_context_key = metrics_context_key
            metrics_ready = True

        if not selected_metric_options:
            st.info("No metrics selected.")
        elif metrics_ready:
            _render_stretch_dataframe(st, st.session_state.metric_rows)

    active_step = _quick_active_step(
        audio_loaded,
        message_valid,
        encoded_ready,
        attack_ready,
        decoded_ready,
        metrics_ready or (decoded_ready and attack_ready and not selected_metric_options),
    )
    step_counter.caption(f"Step {active_step}/{total_steps}")
    step_progress.progress(active_step / total_steps)


def _render_encoded_summary(st, audio_samples: np.ndarray) -> None:
    st.subheader("Encoded file")
    st.write(f"Method: {st.session_state.selected_method}")
    st.write(f"Message length: {len(st.session_state.original_message_bits)} bits")
    st.write("Original message")
    st.code(bits_to_string(st.session_state.original_message_bits))
    st.write(f"Sample rate: {st.session_state.sample_rate} Hz")
    st.write(f"Samples: {audio_samples.shape[0]}")


def _render_decode_result(st, original_bits: list[int], decoded_bits: list[int]) -> None:
    correct_bits = calculate_correct_bits(original_bits, decoded_bits)
    accuracy = calculate_bit_accuracy(original_bits, decoded_bits)
    decoded_text = bits_to_string(decoded_bits)
    matches = decoded_bits == original_bits

    st.subheader("Decoding result")
    st.write("Decoded message")
    st.code(decoded_text)
    st.write("Match")
    if matches:
        st.success("Messages match.")
    else:
        st.error("Messages do not match.")
    st.write("Bit accuracy")
    st.write(f"{correct_bits} / {len(original_bits)} ({accuracy:.2%})")


def _calculate_selected_metrics(
    selected_metric_options,
    original_samples: np.ndarray,
    encoded_samples: np.ndarray,
    sample_rate: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for option in selected_metric_options:
        try:
            metric = create_metric(option.metric_type)
            value = metric.calculate(original_samples, encoded_samples, sample_rate)
        except Exception as exc:
            rows.append(
                {
                    "name": option.name,
                    "class": option.class_name,
                    "category": option.category,
                    "value": str(exc),
                    "status": "error",
                }
            )
        else:
            rows.append(
                {
                    "name": option.name,
                    "class": option.class_name,
                    "category": option.category,
                    "value": _format_metric_value(value),
                    "status": "ok",
                }
            )
    return rows


def _format_metric_value(value) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (int, float, np.integer, np.floating)):
            return f"{float(scalar):.6g}"
        return str(scalar)
    return np.array2string(array, precision=6, threshold=12)


def _quick_context_key(uploaded_file, sample_rate: int, selected_option, message_bits: list[int]) -> tuple:
    return (
        getattr(uploaded_file, "name", ""),
        getattr(uploaded_file, "size", None),
        sample_rate,
        selected_option.method_type.name,
        bits_to_string(message_bits),
    )


def _quick_active_step(
    audio_loaded: bool,
    message_valid: bool,
    encoded_ready: bool,
    attack_ready: bool,
    decoded_ready: bool,
    metrics_ready: bool,
) -> int:
    if metrics_ready:
        return 7
    if decoded_ready:
        return 6
    if attack_ready:
        return 5
    if encoded_ready:
        return 4
    if message_valid:
        return 3
    if audio_loaded:
        return 2
    return 1


def _ensure_generated_message(st, message_length: int) -> None:
    if (
        "generated_message_bits" not in st.session_state
        or st.session_state.get("generated_message_length") != message_length
    ):
        st.session_state.generated_message_bits = generate_message(message_length)
        st.session_state.generated_message_length = message_length


def _debug_exception(st, exc: Exception) -> None:
    with st.expander("Technical details"):
        st.exception(exc)


if __name__ == "__main__":
    if _is_running_with_streamlit():
        run_app()
    else:
        main()
    