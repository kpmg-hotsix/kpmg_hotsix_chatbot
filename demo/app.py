import gradio as gr
import re
from utils import *


def process_query(t):
    if "사용법" in t:
        return '''
            다음 예시 중에서 한 가지 택해서 참고하여 입력해주세요. \n
            
            "자연어 처리 기술과 관련된 기업 알려줘" \n
            "매스프레소와(과) 유사한 기업 찾아줘" \n
            "매스프레소의 최근 이슈 알려줘" \n
            "매스프레소의 재무제표 알려줘" \n
            '''

    if "기술과" in t:
        company = re.split(' 기술과', t)[0]
        tech_company = tech_search(company)
        return '\n'.join(tech_company)

    if "유사한 기업" in t:
        company = re.split('[과|와]', t)[0] #'과' 말고도 '와'도 경우의 수에 들어가므로..
        typo_company = Typo_correction(company)
        similar_company = similar_companies(typo_company)
        return '\n'.join(similar_company[0])

    if "최근 이슈" in t:
        company = re.split('의', t)[0]
        typo_company = Typo_correction(company)
        news_sentiment = find_sentiment(typo_company)
        return news_sentiment

                
    if "재무제표" in t:
        company = re.split('의', t)[0]
        fin = find_finance(company)
        return f'''
        유동비율: {fin['유동비율'].values[0]} \n
        자기자본비율: {fin['자기자본비율'].values[0]} \n
        부채비율: {fin['부채비율'].values[0]} \n
        총자산회전율: {fin['총자산회전율'].values[0]} \n
        총자산증가율: {fin['총자산증가율'].values[0]} \n
        매출액증가율: {fin['매출액증가율'].values[0]} \n
        순이익증가율: {fin['순이익증가율'].values[0]} \n

        (기준년도: {fin['기준년도'].values[0]})
        '''

# import plotly.graph_objects as go

# finance = pd.read_csv('./data/finance.csv')
# categories = finance.columns.tolist()[1:-1]

# fig = go.Figure()

# first=11; second=12

# fig.add_trace(go.Scatterpolar(
#     r=finance.iloc[first].values[1:-1],
#     theta=categories,
#     name=finance['기업명'][first],
#     fill='toself',
# ))
# fig.add_trace(go.Scatterpolar(
#     r=finance.iloc[second].values[1:-1],
#     theta=categories,
#     name=finance['기업명'][second],
#     fill='toself',
# ))

# fig.show()


def response(text, state):
    state = state + [(text, process_query(text))]
    return state, state


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            state = gr.State([])
            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="메세지를 입력해주세요").style(container=False)
                    
            txt.submit(response, [txt, state], [chatbot, state])
        with gr.Column():
            with gr.Box():
                gr.Markdown("## Select graphs to display")
            
demo.launch(debug=True)