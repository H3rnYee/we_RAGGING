from __future__ import annotations

import streamlit as st

from optimizer.optimizer_prompt import optimizer_template
from retriever.core import (
    build_retrieved_context,
    stream_optimized_prompt,
    get_all_tags,
)
from retriever.core import load_prompts, search_prompts
from evaluation.scorer import score_optimization


st.set_page_config(page_title="Prompt RAG Optimizer", layout="wide")


@st.cache_data
def cached_prompts() -> list[dict]:
    return load_prompts()


def main() -> None:
    prompts = cached_prompts()
    all_tags = get_all_tags(prompts)

    st.title("Prompt RAG Optimizer")

    with st.sidebar:
        st.header("Retrieval")
        selected_tags = st.multiselect("Boost tags", all_tags)
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)
        tag_weight = st.slider(
            "Tag boost strength",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )

    user_prompt = st.text_area(
        "Prompt",
        height=160,
        placeholder="Describe the coding task you want to improve...",
    )

    run_retrieval = st.button("Retrieve and optimize", type="primary")
    if not run_retrieval:
        return

    if not user_prompt.strip():
        st.warning("Enter a prompt first.")
        return

    try:
        results = search_prompts(
            user_prompt,
            tags=selected_tags,
            k=top_k,
            tag_weight=tag_weight,
            prompts=prompts,
        )
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    retrieved_context = build_retrieved_context(results)

    strategies_tab, optimized_tab, eval_tab = st.tabs(
        ["Retrieved strategies", "Optimized prompt", "Evaluation"]
    )

    with strategies_tab:
        for result in results:
            prompt = result.prompt
            tags = ", ".join(prompt.get("tags", []))
            matched = ", ".join(result.matched_tags) or "none"

            with st.container(border=True):
                st.markdown(f"**{prompt.get('id', 'unknown')}**")
                st.write(prompt["content"])
                st.caption(
                    f"Boosted score {result.boosted_score:.3f} | "
                    f"Similarity {result.similarity:.3f} | "
                    f"Tag boost {result.tag_score:.3f} | "
                    f"Matched tags: {matched} | Tags: {tags}"
                )

    with optimized_tab:
        try:
            stream = stream_optimized_prompt(
                user_prompt,
                retrieved_context,
                optimizer_template,
            )
            optimized_prompt = st.write_stream(stream)
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
            st.stop()

        if optimized_prompt:
            st.session_state["last_optimized"] = optimized_prompt
            st.session_state["last_original"] = user_prompt

    with eval_tab:
        original = st.session_state.get("last_original", user_prompt)
        optimized = st.session_state.get("last_optimized")

        if not optimized:
            st.info("Generate an optimized prompt first (Optimized prompt tab).")
        else:
            st.markdown("**Original**")
            st.text(original)
            st.markdown("**Optimized**")
            st.text(optimized)

            if st.button("Run evaluation", key="eval_btn"):
                with st.spinner("Scoring..."):
                    try:
                        report = score_optimization(original, optimized)
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        st.stop()

                col1, col2, col3 = st.columns(3)
                col1.metric("Specificity", f"{report['specificity']}/5")
                col2.metric("Actionability", f"{report['actionability']}/5")
                col3.metric("Faithfulness", f"{report['faithfulness']}/5")

                st.metric(
                    "Heuristic clarity delta",
                    f"{report['heuristics']['clarity_delta']:+.1f}%",
                    help="Change in average word length (proxy for precision vs. vagueness)",
                )
                st.metric(
                    "Length ratio",
                    f"{report['heuristics']['length_ratio']:.2f}×",
                    help="Tokens in optimized prompt / tokens in original",
                )

                if report.get("reasoning"):
                    with st.expander("Judge reasoning"):
                        st.write(report["reasoning"])


if __name__ == "__main__":
    main()
