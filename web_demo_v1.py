import json
import streamlit as st
import openai

st.set_page_config(page_title="Baichuan 2")
st.title("Baichuan 2")


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []
    return st.session_state.messages


def main():
    openai.api_base = "http://localhost:5000/v1"
    openai.api_key = "none"

    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            answer = ""
            for response in openai.ChatCompletion.create(
                    model="baichuan2",
                    messages=messages,
                    stream=True):
                if hasattr(response.choices[0].delta, "content"):
                    answer = answer + response.choices[0].delta.content
                    placeholder.markdown(answer)

        messages.append({"role": "assistant", "content": answer})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
