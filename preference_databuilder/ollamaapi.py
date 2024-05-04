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
    response = client.chat.completions.create(model="qwen:0.5b", messages=messages, stream=True)
    answer = ''
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token != None:
            answer += token
            print(token, end='')

    messages.append({"role": "assistant", "content": answer})
    print()
    
#测试模型同一prompt的回复
# for _ in range(10):
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the meaning of life?"},
#     ]
#     response = client.chat.completions.create(model="phi3", messages=messages, stream=True)
#     for chunk in response:
#         token = chunk.choices[0].delta.content
#         if token != None:
#             print(token, end='')
#     print()
    """
    这段代码的实现原理是通过遍历API响应中的每个chunk，
    并从每个chunk的choices列表中提取第一个delta对象的content属性。
    然后，如果content属性不为空，则将其打印到控制台，并在打印时将end参数设置为''，
    以便在每次打印后不会换行。最后，当循环结束后，打印一个换行符以结束输出。
    """
