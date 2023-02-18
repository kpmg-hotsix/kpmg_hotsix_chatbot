import gradio as gr
import re
from utils import *

# 챗봇 쿼리 처리
def process_query(t):
    if "사용법" in t:
        return '''
            다음 예시 중에서 한 가지 택해서 참고하여 입력해주세요. \n
            
            자연어 처리 기술과 관련된 기업 알려줘\n
            매스프레소와(과) 유사한 기업 찾아줘\n
            매스프레소의 최근 이슈 알려줘\n
            매스프레소의 재무제표 알려줘\n
            '''

    if "기술과" in t:
        company = re.split(' 기술과', t)[0]
        tech_company = tech_search(company)
        return '\n'.join(tech_company)

    if "유사한 기업" in t:
        company = re.split('[과|와]', t)[0] #'과' 말고도 '와'도 경우의 수에 들어가므로..
        typo_company = typo_correction(company)
        similar_company = similar_companies(typo_company)
        return '\n'.join(similar_company[0])

    if "최근 이슈" in t:
        company = re.split('의', t)[0]
        typo_company = typo_correction(company)
        news_sentiment = find_sentiment(typo_company)
        results = []
        for title, result in zip(news_sentiment['text'], news_sentiment['result']):
            results.append(f'{title}: {result}')
        
        return '\n\n'.join(results)

                
    if "재무제표" in t:
        company = re.split('의', t)[0]
        fin = find_finance(company)
        return f'''
        유동비율: {fin['유동비율'].values[0]}\n
        자기자본비율: {fin['자기자본비율'].values[0]}\n
        부채비율: {fin['부채비율'].values[0]}\n
        총자산회전율: {fin['총자산회전율'].values[0]}\n
        총자산증가율: {fin['총자산증가율'].values[0]}\n
        매출액증가율: {fin['매출액증가율'].values[0]}\n
        순이익증가율: {fin['순이익증가율'].values[0]} \n

        (기준년도: {fin['기준년도'].values[0]})
        '''

    else:
        return "죄송합니다. 무슨 말씀이신지 이해하지 못했습니다."


# 챗봇 히스토리 저장
def response(text, state):
    state = state + [(text, process_query(text).replace('\n','<br>'))]
    return state, state


with gr.Blocks(title='Hotsix', css='''.gradio-container {height: 700px}''') as demo:
    # with gr.Row().style(equal_height=True):
    
    gr.HTML('''
        <div class="content">
            <img src="file/image/logo.png">
            <h3><strong>기술 키워드 기반 기업 검색 시스템</strong></h3>
        </div>
        <style>
            img {
                float: right;
                width: 70px;
                height: 70px;
                object-fit: cover;
            }
            p {
                float: left;
            }
            div {
                
                vertical-align: middle;
            }
        </style>
    ''')
    # gr.Markdown('''## 기술 키워드 기반 기업 검색 시스템  ![image](./image/logo.jpeg){: style="float: right"}''')
    # 챗봇 인터페이스
    with gr.Column(variant='panel'):
        chatbot = gr.Chatbot(show_label=False)
        state = gr.State([])
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="메세지를 입력해주세요").style(container=False)
                
        txt.submit(response, [txt, state], [chatbot, state])

        # 결과 창 인터페이스
        # with gr.Column():
        #     with gr.Box():
        #         gr.Markdown("## Select graphs to display")
            
demo.launch(favicon_path='file/image/logo.ico', inbrowser=True)