from openai import OpenAI

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    
    base_url="http://localhost:11434/v1/"
)


while True:
    user_input = input('User:')
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(model="phi3", messages=messages, stream=True)
    answer = ''
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token != None:
            answer += token
            print(token, end='')

    messages.append({"role": "assistant", "content": answer})
    print()
    
