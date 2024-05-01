
"""
这是注释
"""
import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from transformers import (
    AutoModel,
    AutoTokenizer,

)
import json
from openai import OpenAI

from dotenv import load_dotenv
# 在使用API密钥和基础URL之前加载.env文件
load_dotenv()

# 现在可以通过os.environ访问这些值
API_BASE = os.environ.get("API_BASE")
API_KEY = os.environ.get("API_KEY")


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=API_KEY,
    base_url=API_BASE
)







st.set_page_config(
    page_title="Rank List Labeler",
    page_icon='📌',
    layout="wide"
)

MODEL_CONFIG = {
    'model_name': 'THUDM/chatglm-6b-int4',             # backbone
    'dataset_file': 'total_dataset.json',       # 标注数据集的存放文件
    'rank_list_len': 6,                                           # 排序列表的长度
    'max_gen_seq_len': 1000,                                        # 生成答案最大长度
    'random_prompts': [                                           # 随机prompt池
                        '今天我去了',
                        '这部电影很',
                        '刚收到货，感觉',
                        '这部电影很',
                        '说实话，真的很',
                        '这次购物总的来说体验很'
                    ]
}


######################## 页面配置初始化 ###########################
RANK_COLOR = [
    'red',
    'green',
    'blue',
    'orange',
    'violet'
]


######################## 会话缓存初始化 ###########################
if 'model_config' not in st.session_state:
    st.session_state['model_config'] = MODEL_CONFIG

@st.cache_resource()



def predict(input, history=None):
    completion = client.chat.completions.create(
    model="yi-34b-chat-200k",
    messages=[{"role": "user", "content":input}],
    max_tokens=6000,
    top_p=0.8,
    # stream=True,
)
    outputtext=completion.choices[0].message.content
    print(outputtext)
    with st.empty():
        st.write(input) # 将 message 函数改成了 st.write
        st.write("AI正在回复:")
        for response, history in model.stream_chat(tokenizer, input, history):
            query, response = history[-1]
            st.write(response)
        return response



if 'current_results' not in st.session_state:
    st.session_state['current_results'] = [''] * MODEL_CONFIG['rank_list_len']

if 'current_prompt' not in st.session_state:
    st.session_state['current_prompt'] = '今天早晨我去了'


######################### 函数定义区 ##############################

def generate_text():
    current_results = []
    for _ in range(MODEL_CONFIG['rank_list_len']):

        # 检查是否被中断
        if 'interrupt_flag' in st.session_state and st.session_state['interrupt_flag']:
            st.session_state['interrupt_flag'] = False  # 清除中断标志
            break

        # 生成文本
        result = predict(st.session_state['current_prompt'])
        generated_text = result
        if len(generated_text) > MODEL_CONFIG['max_gen_seq_len']:
            generated_text = generated_text[:MODEL_CONFIG['max_gen_seq_len']]
        current_results.append(generated_text)
        if len(current_results) == MODEL_CONFIG['rank_list_len']:
            break  # 列表长度已达到最大值，跳出循环
    st.session_state['current_results'] = current_results
    print(current_results)
    return current_results


# 存储当前排序到json文件中
def save_to_json(data):
    with open(MODEL_CONFIG["dataset_file"], "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

# 从json文件中读取之前标注过的数据集
def read_from_json():
    rank_texts_list = []
    if not os.path.exists(MODEL_CONFIG["dataset_file"]):
        st.warning("暂无数据，请开始标注")
    else:
        with open(MODEL_CONFIG["dataset_file"], "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                rank_texts_list.append(data)
    return rank_texts_list
######################### 页面定义区（侧边栏） ########################
st.sidebar.title('让Yi大模型自动给予训练模型的生成进行排序')
# st.sidebar.markdown('''
#     ```
#                     
#     ```
# ''')
st.sidebar.markdown('''
    
                    简单来说就是你经过sft微调后，想通过RLHF（ppo奖励模型）训练怎么样的模型，就让Yi大模型自动给你的模型生成回答进行排序，最后再导出偏好数据去训练奖励模型。
    
''')
st.sidebar.markdown('本项目开源在[github](https://arxiv.org/pdf/2203.02155.pdf) 。')


label_tab, dataset_tab = st.tabs(['Label', 'Dataset'])


######################### 页面定义区（标注页面） ########################
with label_tab:
    with st.expander('🔍 Setting Prompts', expanded=True):
        random_button = st.button('随机 prompt',
                                  help='从prompt池中随机选择一个prompt，可通过修改源码中 MODEL_CONFIG["random_prompts"] 参数来自定义prompt池。')
        if random_button:
            prompt_text = random.choice(MODEL_CONFIG['random_prompts'])
        else:
            prompt_text = st.session_state.get('current_prompt', '')

        query_txt = st.text_input('prompt: ', prompt_text)
        if query_txt != st.session_state.get('current_prompt', ''):
            st.session_state['current_prompt'] = query_txt
            generate_text()

        interrupt_button = st.button('中断输出', key='interrupt_button', help='按下中断输出按钮可中断生成答案的操作。')
        if interrupt_button:
            st.session_state['interrupt_flag'] = True

        generate_button = st.button('生成结果', key='generate_button', help='按下 Enter 键来生成结果。')
        if generate_button:
            generate_text()

    with st.expander('💡 Generate Results', expanded=True):
        columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
        rank_results = [-1] * MODEL_CONFIG['rank_list_len']
        rank_choices = [-1] + [i + 1 for i in range(MODEL_CONFIG['rank_list_len'])]
        for i, c in enumerate(columns):
            with c:
                choice = st.selectbox(f'句子{i + 1}排名', rank_choices,
                                      help='为当前句子选择排名，排名越小，得分越高。（-1代表当前句子暂未设置排名）')
                if choice != -1 and choice in rank_results:
                    st.info(
                        f'当前排名[{choice}]已经被句子[{rank_results.index(choice) + 1}]占用，请先将占用排名的句子置为-1再为当前句子分配该排名。')
                else:
                    rank_results[i] = choice
                color = RANK_COLOR[i] if i < len(RANK_COLOR) else 'white'
                current_results = st.session_state.get('current_results', [])
                if i < len(current_results):
                    text = current_results[i]
                else:
                    text = ''

                st.markdown(f":{color}[{text}]")

    with st.expander('🥇 Rank Results', expanded=True):
        columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
        for i, c in enumerate(columns):
            with c:
                st.write(f'Rank {i+1}：')
                if i + 1 in rank_results:
                    color = RANK_COLOR[rank_results.index(i+1)] if rank_results.index(i+1) < len(RANK_COLOR) else 'white'
                    st.markdown(f":{color}[{st.session_state['current_results'][rank_results.index(i+1)]}]")

    # 替换页面定义区标注页面中的数据读取方式和保存方式
    save_button = st.button('存储当前排序')
    if save_button:
        if -1 in rank_results:
            st.error('请完成排序后再存储！', icon='🚨')
            st.stop()

        prompt_text = st.session_state['current_prompt']
        data = {"prompt": prompt_text}
        for i in range(len(rank_results)):
            data[f'rank_{i + 1}'] = st.session_state['current_results'][rank_results.index(i + 1)]
        save_to_json(data)

        st.success('保存成功，请更换prompt生成新的答案~', icon="✅")

######################### 页面定义区（数据集页面） #######################
# 替换页面定义区数据集页面中的数据读取方式
with dataset_tab:
# 读取数据并创建 DataFrame 示例
    rank_texts_list = read_from_json()
    df = pd.DataFrame(rank_texts_list)

    # 在 Streamlit 界面中显示可编辑表格
    edited_df = st.data_editor(df)

# 增加保存按钮
    if st.button('保存更改'):
        # 将编辑后的 DataFrame 转换为 list 格式
        updated_data = edited_df.to_dict('records')

        with open(MODEL_CONFIG["dataset_file"], "w", encoding="utf-8") as f:
            for data in updated_data:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

    st.success('保存成功！')

