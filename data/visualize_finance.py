import plotly.graph_objects as go

finance = pd.read_csv('./data/finance.csv')
categories = finance.columns.tolist()[1:-1]

fig = go.Figure()

first=11; second=12

fig.add_trace(go.Scatterpolar(
      r=finance.iloc[first].values[1:-1],
      theta=categories,
      name=finance['기업명'][first],
      fill='toself',
))
fig.add_trace(go.Scatterpolar(
      r=finance.iloc[second].values[1:-1],
      theta=categories,
      name=finance['기업명'][second],
      fill='toself',
))

fig.show()