import hashlib

import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

import config
from ai_tools import ChatTools, ToolError
from ai_agent import ChatAgent, friendly_api_error
from retrieval import MessageIndex

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")

st.markdown("""
    <style>
        /* ---- section headings inside the dashboard ---- */
        .styled-heading {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 2.2rem 0 0.8rem 0;
        }
        /* NOTE: no negative margin here. The old "-3rem" rule pulled the first heading
           up over the tab bar and blocked the tabs from being clicked. */
        .styled-heading.first-heading { margin-top: 0.6rem; }
        .blue   { color: #3366cc; }
        .green  { color: #2e8b57; }
        .red    { color: #cc3300; }
        .purple { color: #800080; }
        .orange { color: #ff6600; }
        .pink   { color: #cc3399; }

        /* ---- welcome box shown before upload ---- */
        .info-box {
            background-color: #f0f8ff;
            border-left: 6px solid #1e90ff;
            padding: 1rem 1.5rem;
            margin: 1rem 0 2.5rem 0;
            border-radius: 8px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1rem;
            color: #003366;
        }

        /* ---- tab bar: bigger, well-spaced, clickable exactly where the label is ---- */
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.6rem 0.4rem;
        }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

        /* ---- stat cards ---- */
        .stat-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .stat-label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.2rem; }
        .stat-value { font-size: 1.8rem; font-weight: 700; color: #3366cc; }

        /* ---- AI Chat Assistant tab ---- */
        .ai-title {
            font-size: 1.8rem; font-weight: 700; color: #a259c4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0.2rem 0 0.2rem 0;
        }
        .ai-subtitle { font-size: 0.95rem; opacity: 0.8; margin-bottom: 0.6rem; }
        .ai-examples-label { font-size: 0.85rem; opacity: 0.7; margin: 0.6rem 0 0.2rem 0; }
        .ai-empty { text-align: center; opacity: 0.6; padding: 2rem 0 1rem 0; font-style: italic; }
        [data-testid="stChatMessage"] { border-radius: 12px; padding: 0.6rem 0.9rem; margin-bottom: 0.4rem; }
        [data-testid="stChatMessage"] pre { font-size: 0.78rem; }
    </style>
""", unsafe_allow_html=True)


def styled_heading(text, color="blue", is_first=False):
    extra_class = "first-heading" if is_first else ""
    st.markdown(f"<div class='styled-heading {color} {extra_class}'>{text}</div>", unsafe_allow_html=True)


def stat_card(col, label, value):
    with col:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>{label}</div>"
                    f"<div class='stat-value'>{value:,}</div></div>", unsafe_allow_html=True)


def show_fig(fig):
    """Render a matplotlib figure and free its memory (Streamlit reruns often)."""
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# File loading (shared by the dashboard and the AI assistant)
# ---------------------------------------------------------------------------
def decode_chat_bytes(raw: bytes) -> str:
    """WhatsApp exports are UTF-8, but Windows tools sometimes re-save them as UTF-16 or cp1252."""
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@st.cache_data(show_spinner="Parsing chat…")
def load_chat(raw: bytes):
    """Decode + preprocess once per uploaded file. Returns (df, None) or (None, error_message)."""
    if not raw or not raw.strip():
        return None, "The uploaded file is empty."
    try:
        df = preprocessor.preprocess(decode_chat_bytes(raw))
    except ValueError as e:
        return None, f"Could not parse this file as a WhatsApp export: {e}"
    except Exception as e:
        return None, f"Unexpected error while parsing the chat: {type(e).__name__}: {e}"
    if df.empty or df["date"].notna().sum() == 0:
        return None, "No messages found. Make sure this is a WhatsApp chat export in 12-hour or 24-hour format."
    return df, None


@st.cache_resource(show_spinner="Preparing AI assistant…")
def build_ai_resources(file_hash: str, _df):
    """Build the tool layer and the semantic index once per uploaded file (keyed by content hash)."""
    tools = ChatTools(_df)
    if int(tools.df["is_text"].sum()) <= config.MAX_INDEXED_MESSAGES:
        tools.index = MessageIndex(tools.df, tools.stopwords)
    return tools


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💬 WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")
st.session_state["uploaded_file"] = uploaded_file

if not uploaded_file:
    st.markdown("<h2 style='margin-top:0'>Welcome 👋</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-box'>
            <strong>How to use:</strong><br>
            1. In WhatsApp open a chat → <em>More</em> → <em>Export chat</em> → <em>Without media</em>.<br>
            2. Upload the exported <strong>.txt</strong> file from the sidebar.<br>
            3. Explore the <strong>Analysis Dashboard</strong>, or ask questions in the
               <strong>AI Chat Assistant</strong> tab.<br><br>
            Both 12-hour <em>(11:15 AM)</em> and 24-hour <em>(23:15)</em> exports are supported;
            day-first and month-first dates are detected automatically.
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df, load_error = load_chat(bytes_data)
    if load_error:
        st.error(load_error)
        st.stop()
    unparsed = int(df["date"].isna().sum())
    if unparsed:
        st.warning(f"{unparsed} of {len(df)} messages have timestamps that could not be parsed and will be "
                   "ignored by the AI assistant. The export may mix date formats.")

    user_list = df["user"].unique().tolist()
    if "group_notification" in user_list:      # a 1-to-1 chat may have no system messages
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    # Remember the click so the dashboard survives reruns caused by the AI chat input.
    if st.sidebar.button("Show Analysis", type="primary", use_container_width=True):
        st.session_state["show_analysis"] = True
    st.sidebar.caption(f"{len(user_list) - 1} participants · {int(df['date'].notna().sum()):,} messages")

    tab_dashboard, tab_ai = st.tabs(["📊 Analysis Dashboard", "🤖 AI Chat Assistant"])

    # =====================================================================
    # Analysis Dashboard
    # =====================================================================
    with tab_dashboard:
        if not st.session_state.get("show_analysis"):
            st.info("Select a user in the sidebar and click **Show Analysis** to build the dashboard.")
        else:
            # --- Top Stats ---
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            styled_heading(f"Top Statistics — {selected_user}", "blue", is_first=True)
            c1, c2, c3, c4 = st.columns(4)
            stat_card(c1, "Total Messages", num_messages)
            stat_card(c2, "Total Words", words)
            stat_card(c3, "Media Shared", num_media_messages)
            stat_card(c4, "Links Shared", num_links)

            # --- Most Busy Users ---
            if selected_user == "Overall":
                styled_heading("Most Busy Users", "pink")
                x, new_df = helper.most_busy_users(df)
                col1, col2 = st.columns([3, 2])
                with col1:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color="#cc3399")
                    ax.set_ylabel("Messages")
                    plt.xticks(rotation=45, ha="right")
                    show_fig(fig)
                with col2:
                    st.dataframe(new_df, use_container_width=True, hide_index=True)

            # --- Wordcloud ---
            styled_heading("Wordcloud", "green")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(df_wc)
            ax.axis("off")
            show_fig(fig)

            # --- Most Common Words ---
            styled_heading("Most Common Words", "red")
            most_common_df = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(most_common_df[0], most_common_df[1], color="#cc3300")
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            show_fig(fig)

            # --- Emoji Analysis ---
            styled_heading("Emoji Analysis", "purple")
            emoji_df = helper.emoji_helper(selected_user, df)
            if emoji_df.empty:
                st.info("No emojis found for this selection.")
            else:
                col1, col2 = st.columns([3, 4])
                with col1:
                    st.subheader("Emoji Frequency Table")
                    st.dataframe(emoji_df, use_container_width=True, hide_index=True)
                with col2:
                    st.subheader("Emoji Usage Distribution")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(emoji_df["Count"].head(), labels=emoji_df["Emoji"].head(), autopct="%0.2f%%")
                    show_fig(fig)

            # --- Activity Map ---
            styled_heading("Activity Map", "orange")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color="purple")
                plt.xticks(rotation=45, ha="right")
                show_fig(fig)
            with col2:
                st.subheader("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color="orange")
                plt.xticks(rotation=45, ha="right")
                show_fig(fig)

            # --- Weekly Activity Map ---
            styled_heading("Weekly Activity Map", "blue")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(user_heatmap, ax=ax, cmap="viridis")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("")
            show_fig(fig)

            # --- Monthly Timeline ---
            styled_heading("Monthly Timeline", "green")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(timeline["time"], timeline["message"], color="green", marker="o")
            ax.set_ylabel("Messages")
            plt.xticks(rotation=45, ha="right")
            show_fig(fig)

            # --- Daily Timeline ---
            styled_heading("Daily Timeline", "red")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily_timeline["only_date"], daily_timeline["message"], color="black")
            ax.set_ylabel("Messages")
            plt.xticks(rotation=45, ha="right")
            show_fig(fig)

    # =====================================================================
    # AI Chat Assistant
    # =====================================================================
    with tab_ai:
        # -- state: reset the conversation whenever a different file is uploaded
        file_hash = hashlib.sha256(bytes_data).hexdigest()
        if st.session_state.get("ai_file_hash") != file_hash:
            st.session_state["ai_file_hash"] = file_hash
            st.session_state["ai_messages"] = []
        messages = st.session_state["ai_messages"]

        # -- header row: title + actions
        head_col, act_col = st.columns([4, 1.4])
        with head_col:
            st.markdown("<div class='ai-title'>🤖 AI Chat Assistant</div>"
                        "<div class='ai-subtitle'>Ask anything about this chat. Every number and quote is computed "
                        "from the uploaded file — only the question and the small result go to the LLM.</div>",
                        unsafe_allow_html=True)
        with act_col:
            if messages:
                transcript = "\n\n".join(f"{'You' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in messages)
                st.download_button("⬇️ Save chat", transcript, file_name="ai_chat_history.txt",
                                   use_container_width=True)
                if st.button("🗑️ Clear chat", use_container_width=True):
                    st.session_state["ai_messages"] = []
                    st.rerun()

        api_key = config.get_api_key()
        if not api_key:
            st.warning("**GROQ_API_KEY is not set.** Add it to a local `.env` file (`GROQ_API_KEY=gsk_...`) "
                       "or to Streamlit secrets, then restart the app. See README.md.")
            st.stop()
        try:
            tools = build_ai_resources(file_hash, df)
        except ToolError as e:
            st.error(str(e))
            st.stop()
        if tools.index is None:
            st.info("This chat is very large, so topic/semantic questions are disabled; "
                    "counts, rankings and keyword search still work.")

        # -- example questions (one row of pills; falls back to buttons on old Streamlit)
        examples = [
            "Who is the most active user?", "What was the busiest day?", "What is this chat mainly about?",
            "What time is this chat most active?", "Who shares more media?", "What was the first message?",
        ]

        def _pick_example():
            st.session_state["ai_pending"] = st.session_state.get("ai_example")
            st.session_state["ai_example"] = None      # allow the same pill to be clicked again later

        st.markdown("<div class='ai-examples-label'>Try one of these</div>", unsafe_allow_html=True)
        if hasattr(st, "pills"):
            st.pills("examples", examples, key="ai_example", on_change=_pick_example,
                     label_visibility="collapsed")
        else:
            for row in (examples[:3], examples[3:]):
                for col, q in zip(st.columns(3), row):
                    if col.button(q, use_container_width=True, key=f"ex_{q}"):
                        st.session_state["ai_pending"] = q

        st.divider()

        # -- conversation (always rendered from session state, oldest -> newest)
        if not messages:
            st.markdown("<div class='ai-empty'>No questions yet. Pick an example above or type below.</div>",
                        unsafe_allow_html=True)
        for m in messages:
            with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
                st.markdown(m["content"])
                if m.get("trace"):
                    with st.expander("How was this computed?"):
                        for step in m["trace"]:
                            st.markdown(f"**{step['tool']}** `{step['input']}`")
                            st.code(step["output"][:1500], language="json")
        pending_slot = st.container()   # the in-progress answer renders here, in flow

        # -- input
        typed = st.chat_input("Ask a question about this chat…")
        question = (typed or st.session_state.pop("ai_pending", None) or "").strip()

        if question:
            if len(question) > 1000:
                st.warning("Please keep questions under 1000 characters.")
                st.stop()
            history = [{"role": m["role"], "content": m["content"]} for m in messages]
            with pending_slot:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analysing the chat…"):
                        try:
                            answer, trace = ChatAgent(tools, api_key=api_key).ask(question, history)
                        except Exception as e:      # API / network / rate-limit -> friendly text, never a crash
                            answer, trace = friendly_api_error(e), []
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer, "trace": trace})
            st.rerun()   # redraw everything from state so the new pair sits in the conversation flow
