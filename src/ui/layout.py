from __future__ import annotations

import streamlit as st

from src.ui.theme import NAV_PAGES
from src.utils.config import AppConfig


LOGO_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
  <rect x="3" y="3" width="7" height="7" rx="1.6" fill="white"/>
  <rect x="14" y="3" width="7" height="7" rx="1.6" fill="white" opacity="0.85"/>
  <rect x="3" y="14" width="7" height="7" rx="1.6" fill="white" opacity="0.85"/>
  <rect x="14" y="14" width="7" height="7" rx="1.6" fill="white"/>
</svg>
"""


def render_sidebar(config: AppConfig) -> str:
    current = st.session_state.get("page", "Dashboard")
    ui = config.ui()
    operator = ui.get("operator_name", "Dr. Alan Vance")
    operator_id = ui.get("operator_id", "LAB-RE_01")

    st.markdown(
        f"""
        <div class="nd-brand">
          <div class="nd-logo">{LOGO_SVG}</div>
          <div>
            <p class="nd-brand-name">NeuroDefect</p>
            <p>Decision System</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
        anchors=False,
    )

    for label, key in NAV_PAGES:
        is_active = current == key
        if st.button(label, key=f"nav_{key}", width="stretch", type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown(
        f"""
        <div class="nd-user">
          <div class="nd-avatar"></div>
          <div>
            <strong>{operator}</strong>
            <span>{operator_id}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
        anchors=False,
    )
    return current
