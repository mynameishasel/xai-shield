"""Game-style elementary learning mode for XAI Shield (grades 3–5).

Educational demo only: fictional scenarios, synthetic model data, no student data stored.
"""
import pandas as pd
import streamlit as st

FEATURES = ["packet_count", "connection_duration", "failed_connections", "bytes_sent", "bytes_received"]
MISSIONS = [
    {"id": "everyday", "title": "The Busy School Computer", "emoji": "🏫", "color": "#2E86DE", "badge": "Careful Observer", "badge_emoji": "🔎", "story": "The school computer is sending ordinary messages during class. Does the AI notice anything unusual?", "values": [400, 30.0, 2, 12000, 18000], "clue": "Only a few connection attempts failed, and the amount of activity is not unusually high in this pretend example.", "question": "What is a sensible next step if the activity looks ordinary?", "choices": ["Keep an eye on it as usual", "Try to guess someone else's password", "Tell everyone there is definitely an attack"], "correct": 0, "why": "Routine monitoring is reasonable for this example. An AI prediction is a clue, not a guarantee."},
    {"id": "login", "title": "The Password Puzzle", "emoji": "🔐", "color": "#8E44AD", "badge": "Password Protector", "badge_emoji": "🔐", "story": "A pretend school account has several failed connection attempts. Let's see what the AI thinks.", "values": [850, 2.0, 9, 35000, 12000], "clue": "Repeated failed connections may be a sign that someone is having trouble logging in, or that something needs checking.", "question": "Who should a student tell about repeated suspicious sign-in attempts?", "choices": ["A stranger online", "A trusted teacher or school technology adult", "Nobody, because computers are always right"], "correct": 1, "why": "A trusted adult can check the situation safely. Students should not try to access other people's accounts."},
    {"id": "transfer", "title": "The Data Delivery Mystery", "emoji": "📦", "color": "#D97706", "badge": "Evidence Explorer", "badge_emoji": "🏅", "story": "A pretend computer sends a lot of information. Is that automatically dangerous?", "values": [4500, 35.0, 5, 850000, 2500], "clue": "A large amount of sent data and several failed connections can be clues worth investigating. Some large transfers are completely legitimate.", "question": "What should we remember when an AI says something looks suspicious?", "choices": ["It proves a hacker is present", "We should immediately delete everything", "We should check more evidence before deciding"], "correct": 2, "why": "AI can make mistakes. A responsible detective checks other evidence before deciding what happened."},
]


def _safe_probability(model, sample):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Student Mode requires a binary model with class 1 for MALICIOUS.")
    return float(model.predict_proba(sample)[0][classes.index(1)])


def _friendly_level(probability):
    if probability < 0.25:
        return "Looks okay", "🟢"
    if probability < 0.50:
        return "Let's look closer", "🟡"
    if probability < 0.75:
        return "Needs checking", "🟠"
    return "Ask for help", "🔴"


def render_student_mode(model):
    """Render a playful mission-based learning experience with the existing fitted model."""
    st.markdown("""
    <style>
      .academy-hero {background:linear-gradient(115deg,#142d56,#39448d 58%,#6d3b8d);border-radius:24px;padding:28px 30px;color:white;margin:4px 0 20px;}
      .academy-hero h1 {color:white!important;font-size:clamp(1.8rem,3.5vw,2.9rem)!important;margin:0 0 8px!important;}
      .academy-hero p {color:#edf2ff;font-size:1.06rem;margin:8px 0;}
      .academy-mascot {font-size:4.4rem;line-height:1.2;display:inline-block;vertical-align:middle;margin-right:14px;}
      .academy-kicker {font-weight:800;letter-spacing:.1em;color:#b8e8ff!important;font-size:.85rem!important;}
      .academy-card {background:#17233b;border:1px solid #425473;border-radius:16px;padding:17px 18px;min-height:155px;color:#f5f8ff;margin:6px 0 10px;}
      .academy-card h3 {color:#fff!important;margin:5px 0 10px!important;font-size:1.15rem!important;}
      .academy-card p {color:#d5e1f7!important;font-size:.93rem;}
      .academy-complete {background:linear-gradient(120deg,#164e42,#275d77);padding:22px;border-radius:20px;border:2px solid #65dcb0;color:white;text-align:center;margin:18px 0;}
      .academy-complete h2 {color:white!important;}
      .academy-badge {background:#29364c;border:1px solid #526684;border-radius:13px;padding:15px 10px;text-align:center;font-weight:700;color:white;min-height:98px;}
      .academy-badge.locked {opacity:.55;filter:grayscale(.8);}
    </style>
    """, unsafe_allow_html=True)

    if "academy_completed" not in st.session_state:
        st.session_state.academy_completed = []
    if "academy_badges" not in st.session_state:
        st.session_state.academy_badges = []
    if "academy_selected" not in st.session_state:
        st.session_state.academy_selected = 0

    completed = st.session_state.academy_completed
    st.markdown('''<div class="academy-hero"><div class="academy-kicker">WELCOME TO CYBER ACADEMY • GRADES 3–5</div><h1><span class="academy-mascot">🛡️</span>Meet Shieldy, Cyber Guardian!</h1><p>Become a Cyber Detective! Explore computer mysteries, follow clues, and earn three badges.</p><p>🔎 Investigate &nbsp; • &nbsp; 🧠 Think &nbsp; • &nbsp; 🏅 Earn badges</p></div>''', unsafe_allow_html=True)
    st.info("🧑‍🏫 **Safety first:** These are pretend missions. Never test real accounts or school networks. Ask a trusted adult for help.")

    st.markdown("### 🌟 Your detective progress")
    st.progress(len(completed) / len(MISSIONS), text=f"{len(completed)} of {len(MISSIONS)} missions completed")

    st.markdown("### 🗺️ Choose your mission")
    columns = st.columns(3)
    for i, mission in enumerate(MISSIONS):
        with columns[i]:
            status = "✅ Completed" if mission["id"] in completed else "⭐ Ready to explore"
            st.markdown(f'<div class="academy-card" style="border-top:5px solid {mission["color"]}"><div style="font-size:2rem">{mission["emoji"]}</div><h3>Mission {i+1}: {mission["title"]}</h3><p>{status}</p></div>', unsafe_allow_html=True)
            if st.button(f"{'🔄 Revisit' if mission['id'] in completed else '🚀 Play'} Mission {i+1}", key=f"choose_mission_{i}", use_container_width=True, type="primary" if st.session_state.academy_selected == i else "secondary"):
                st.session_state.academy_selected = i
                st.rerun()

    mission = MISSIONS[st.session_state.academy_selected]
    st.divider()
    st.subheader(f"{mission['emoji']} Mission {st.session_state.academy_selected + 1}: {mission['title']}")
    st.write(mission["story"])
    if st.button("🔎 Investigate this mission", key=f"investigate_{mission['id']}", type="primary"):
        st.session_state[f"academy_revealed_{mission['id']}"] = True

    if st.session_state.get(f"academy_revealed_{mission['id']}", False):
        sample = pd.DataFrame([mission["values"]], columns=FEATURES)
        probability = _safe_probability(model, sample)
        predicted = "unusual" if probability >= 0.5 else "ordinary"
        friendly, symbol = _friendly_level(probability)
        st.markdown(f"### {symbol} Shieldy's clue meter: {friendly}")
        st.write(f"Shieldy's demonstration AI thinks this activity looks **{predicted}**. Its estimated chance of 'malicious' activity is **{probability:.1%}**.")
        st.caption("This model learned from made-up examples. Its score is not a real-world safety rating.")
        st.markdown("**💬 Shieldy's clue:**")
        st.write(mission["clue"])
        with st.expander("🔍 Explore the computer clues (optional)"):
            st.write(f"Failed connections: **{int(mission['values'][2])}**")
            st.write(f"Packets: **{int(mission['values'][0]):,}**")
            st.write(f"Information sent: **{int(mission['values'][3]):,} bytes**")
            st.caption("These are story clues, not SHAP or LIME explanations. Advanced Mode contains technical explanations.")
        st.divider()
        st.markdown("### 🧠 Detective challenge")
        answer = st.radio(mission["question"], mission["choices"], index=None, key=f"answer_{mission['id']}")
        if st.button("✨ Check my answer", key=f"check_{mission['id']}"):
            if answer is None:
                st.warning("Pick an answer first, Detective!")
            elif mission["choices"].index(answer) == mission["correct"]:
                st.session_state[f"academy_correct_{mission['id']}"] = True
                if mission["id"] not in st.session_state.academy_completed:
                    st.session_state.academy_completed.append(mission["id"])
                    st.session_state.academy_badges.append(mission["badge"])
                st.rerun()
            else:
                st.session_state[f"academy_correct_{mission['id']}"] = False
                st.info("Good try! Think about the safest choice and try again.")
                st.write(mission["why"])
        if st.session_state.get(f"academy_correct_{mission['id']}", False):
            st.success(f"🎉 Great detective work! You earned the **{mission['badge']}** badge!")
            st.write(mission["why"])

    st.divider()
    st.markdown("### 🏅 Your Detective Badge Collection")
    badge_cols = st.columns(3)
    for i, item in enumerate(MISSIONS):
        earned = item["id"] in st.session_state.academy_completed
        with badge_cols[i]:
            st.markdown(f'<div class="academy-badge {"" if earned else "locked"}"><div style="font-size:2rem">{item["badge_emoji"] if earned else "🔒"}</div>{item["badge"] if earned else "Mystery Badge"}<div style="font-size:.8rem;font-weight:400">{"Unlocked!" if earned else "Complete mission to unlock"}</div></div>', unsafe_allow_html=True)

    if len(st.session_state.academy_completed) == len(MISSIONS):
        st.markdown('<div class="academy-complete"><div style="font-size:3rem">🎉 🏆 🎊</div><h2>Cyber Academy Complete!</h2><p>You solved all three mysteries and earned every badge. Shieldy says: Great job checking clues and thinking safely!</p></div>', unsafe_allow_html=True)
        if st.button("🔄 Play the academy again"):
            for item in MISSIONS:
                for key in (f"academy_revealed_{item['id']}", f"academy_correct_{item['id']}", f"answer_{item['id']}"):
                    st.session_state.pop(key, None)
            st.session_state.academy_completed = []
            st.session_state.academy_badges = []
            st.session_state.academy_selected = 0
            st.rerun()

    st.caption("Learning demo only • No student names or accounts required • Badges and progress are saved only for this Streamlit session.")
