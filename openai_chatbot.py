"""
OpenAI Chat Completions API 기반 Streamlit 챗봇.
배포: Streamlit Community Cloud → Main file: openai_chatbot.py
Secrets: OPENAI_API_KEY (필수), 선택: OPENAI_MODEL
"""

from __future__ import annotations

import os
from typing import Iterator

import streamlit as st
from openai import OpenAI

SYSTEM_PROMPT = (
    "당신은 친절하고 간결한 한국어 비서입니다. "
    "사용자 질문에 정확히 답하고, 불확실하면 그렇게 말합니다."
)

DEFAULT_MODEL = "gpt-4o-mini"


def _get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except (FileNotFoundError, KeyError, TypeError):
        pass
    v = os.environ.get("OPENAI_API_KEY", "").strip()
    return v or None


def _get_default_model() -> str:
    try:
        if "OPENAI_MODEL" in st.secrets:
            m = str(st.secrets["OPENAI_MODEL"]).strip()
            if m:
                return m
    except (FileNotFoundError, KeyError, TypeError):
        pass
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _openai_stream(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def main() -> None:
    st.set_page_config(
        page_title="OpenAI 챗봇",
        page_icon="💬",
        layout="centered",
    )

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "**API 키가 없습니다.**\n\n"
            "- **로컬**: `.streamlit/secrets.toml`에 `OPENAI_API_KEY`를 넣거나, "
            "환경 변수 `OPENAI_API_KEY`를 설정하세요.\n"
            "- **Streamlit Cloud**: 앱 설정 → **Secrets**에 `OPENAI_API_KEY`를 추가하세요."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    default_model = _get_default_model()

    with st.sidebar:
        st.header("설정")
        model = st.text_input(
            "모델",
            value=st.session_state.get("model_name", default_model),
            help="예: gpt-4o-mini, gpt-4o",
        )
        st.session_state["model_name"] = model.strip() or default_model
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
        )
        if st.button("대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Cloud 배포 시 Secrets에 `OPENAI_API_KEY`를 등록하세요. "
            "선택: `OPENAI_MODEL` (기본값 gpt-4o-mini)."
        )

    st.title("💬 OpenAI 챗봇")
    st.caption("OpenAI Chat Completions · 스트리밍 응답")

    client = OpenAI(api_key=api_key)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("메시지를 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[
                {"role": x["role"], "content": x["content"]}
                for x in st.session_state.messages
            ],
        ]

        collected: list[str] = []

        def stream_with_collect() -> Iterator[str]:
            for piece in _openai_stream(
                client,
                model=st.session_state["model_name"],
                messages=api_messages,
                temperature=temperature,
            ):
                collected.append(piece)
                yield piece

        with st.chat_message("assistant"):
            try:
                st.write_stream(stream_with_collect())
            except Exception as e:
                st.error(f"요청 실패: {e}")
                return

        full = "".join(collected)
        if full.strip():
            st.session_state.messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
