
from LLM_handler import ask_chatbot

# Teszteljük a chatbotot
if __name__ == "__main__":
    while True:
        user_input = input("Kérdés: ")
        if user_input.lower() in ["kilépés", "exit", "stop"]:
            break
        content, answer = ask_chatbot(user_input)
        print("\nChatbot válasz:", answer, "\n")
