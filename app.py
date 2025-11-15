import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Import RAG components
from src.utils.supabase_client import SupabaseClient
from src.utils.gemini_client import GeminiClient
from src.rag.retriever import Retriever
from src.rag.generator import ResponseGenerator
from src.rag.query_processor import QueryProcessor
from src.rag.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv(find_dotenv())

# Page config
st.set_page_config(
    page_title="RAG Football Q&A",
    page_icon="⚽",
    layout="wide"
)

# Initialize RAG pipeline
@st.cache_resource
def init_rag_pipeline():
    """Initialize RAG pipeline once and cache it"""
    try:
        supabase = SupabaseClient()
        gemini_client = GeminiClient()

        pipeline = RAGPipeline(
            retriever=Retriever(supabase,gemini_client),
            generator=ResponseGenerator(gemini_client),
            query_processor=QueryProcessor(gemini_client),
        )
        return pipeline
    except Exception as e:
        st.error(f"Lỗi khởi tạo RAG pipeline: {str(e)}")
        return None

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_context' not in st.session_state:
    st.session_state.current_context = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None

def save_evaluation_event(question, answer, context, ground_truth=None):
    """Save Q&A event for evaluation"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'context': context,
        'ground_truth': ground_truth
    }

    # Create data directory if not exists
    Path('data').mkdir(exist_ok=True)

    # Load existing events
    events_file = 'data/evaluation_events.json'
    if os.path.exists(events_file):
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    else:
        events = []

    # Add new event
    events.append(event)

    # Save back to file
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    return True

def process_question(pipeline, question):
    """Process question through RAG pipeline"""
    try:
        result = pipeline(question)
        return result['answer'], result.get('context', [])
    except Exception as e:
        return f"Lỗi xử lý câu hỏi: {str(e)}", []

# Main UI
st.title("⚽ RAG Football Q&A")
st.markdown("Hệ thống hỏi đáp về bóng đá với RAG (Retrieval-Augmented Generation)")

# Sidebar for history and stats
with st.sidebar:
    st.header("📊 Thống kê")

    # Load saved events for stats
    events_file = 'data/evaluation_events.json'
    if os.path.exists(events_file):
        with open(events_file, 'r', encoding='utf-8') as f:
            saved_events = json.load(f)

        st.metric("Tổng câu hỏi đã lưu", len(saved_events))

        # Count events with ground truth
        with_gt = sum(1 for e in saved_events if e.get('ground_truth'))
        st.metric("Có ground truth", with_gt)
    else:
        st.metric("Tổng câu hỏi đã lưu", 0)

    st.divider()

    st.header("📜 Lịch sử hội thoại")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):  # Show last 10
            with st.expander(f"Q{len(st.session_state.history)-i}: {item['question'][:50]}..."):
                st.write(f"**Câu hỏi:** {item['question']}")
                st.write(f"**Trả lời:** {item['answer'][:200]}...")
    else:
        st.info("Chưa có lịch sử hội thoại")

# Main content area
pipeline = init_rag_pipeline()

if pipeline is None:
    st.error("Không thể khởi tạo RAG pipeline. Vui lòng kiểm tra cấu hình.")
else:
    # Question input section
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Ai là cầu thủ ghi nhiều bàn thắng nhất World Cup 2022?",
            key="question_input"
        )

    with col2:
        ask_button = st.button("🔍 Hỏi", type="primary", use_container_width=True)

    # Process question when button clicked
    if ask_button and question:
        with st.spinner("Đang xử lý câu hỏi..."):
            answer, context = process_question(pipeline, question)

            # Save to session state
            st.session_state.current_question = question
            st.session_state.current_answer = answer
            st.session_state.current_context = context

            # Add to history
            st.session_state.history.append({
                'question': question,
                'answer': answer,
                'context': context
            })

    # Display answer section
    if st.session_state.current_answer:
        st.divider()

        # Answer display
        st.subheader("💡 Câu trả lời:")
        st.write(st.session_state.current_answer)

        # Context display (collapsible)
        if st.session_state.current_context:
            with st.expander("📚 Xem context đã sử dụng"):
                for i, ctx in enumerate(st.session_state.current_context, 1):
                    st.markdown(f"**Context {i}:**")
                    st.json(ctx)

        st.divider()

        # Ground truth section for evaluation
        st.subheader("📝 Đánh giá (Tuỳ chọn)")

        col1, col2 = st.columns([3, 1])

        with col1:
            ground_truth = st.text_area(
                "Nhập câu trả lời ground truth (để đánh giá sau):",
                placeholder="Nhập câu trả lời đúng nếu bạn biết...",
                height=100
            )

        with col2:
            save_button = st.button("💾 Lưu để đánh giá", use_container_width=True)

        if save_button:
            if save_evaluation_event(
                st.session_state.current_question,
                st.session_state.current_answer,
                st.session_state.current_context,
                ground_truth if ground_truth else None
            ):
                st.success("✅ Đã lưu event thành công!")
                # Clear ground truth input
                st.rerun()

    # Quick actions
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Làm mới"):
            st.session_state.current_question = None
            st.session_state.current_answer = None
            st.session_state.current_context = None
            st.rerun()

    with col2:
        if st.button("📥 Xuất lịch sử"):
            if st.session_state.history:
                # Create download content
                download_data = json.dumps(
                    st.session_state.history,
                    ensure_ascii=False,
                    indent=2
                )
                st.download_button(
                    label="💾 Tải xuống lịch sử",
                    data=download_data,
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("Không có lịch sử để xuất")

    with col3:
        if st.button("📊 Xem file đánh giá"):
            events_file = 'data/evaluation_events.json'
            if os.path.exists(events_file):
                with open(events_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                st.json(events)
            else:
                st.info("Chưa có dữ liệu đánh giá")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        RAG Football Q&A System | Built with Streamlit 🚀
    </div>
    """,
    unsafe_allow_html=True
)