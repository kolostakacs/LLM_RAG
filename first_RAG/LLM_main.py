
from LLM_handler import ask_chatbot

# Teszteljük a chatbotot
if __name__ == "__main__":
    while True:
        user_input = input("Kérdés: ")
        if user_input.lower() in ["kilépés", "exit", "stop"]:
            break
        print("\nChatbot válasz:", ask_chatbot(user_input), "\n")
