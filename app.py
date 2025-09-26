import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from retriever import retrieve

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Toolkit Document Search", className="mb-4"),
        dbc.Row([
            dbc.Col([
                dcc.Input(id="query-input", type="text",
                          placeholder="Type your question...", debounce=True,
                          style={"width": "100%", "padding": "10px"}),
                html.Br(), html.Br(),
                dbc.Button("Search", id="search-btn", color="primary")
            ], md=8)
        ]),
        html.Hr(),
        html.Div(id="results-area")
    ],
    fluid=True
)

@app.callback(
    Output("results-area", "children"),
    Input("search-btn", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True
)
def search_callback(_, query):
    if not query:
        return dbc.Alert("Please enter a query.", color="warning")

    results = retrieve(query, top_k=5)
    if not results:
        return dbc.Alert("No results found.", color="danger")

    cards = []
    for idx, r in enumerate(results):
        meta = r["metadata"]
        title = meta.get("name") or f"Document {meta.get('document_id')}"
        subtitle = f"Created: {meta.get('create_date') or 'N/A'} | Published: {meta.get('publish_date') or 'N/A'} | Categories: {', '.join(meta.get('categories') or [])}"
        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H5(title, className="mb-0"),
                            html.Small(subtitle, className="text-muted"),
                            dbc.Button(
                                "Expand Full Document",
                                id={"type": "collapse-btn", "index": idx},
                                color="link",
                                n_clicks=0,
                                style={"float": "right"}
                            ),
                            html.Div(f"Score: {r['score']:.3f}", style={"float": "right", "marginRight": "1em"})
                        ],
                        style={"display": "flex", "flexDirection": "column"}
                    ),
                    dbc.CardBody(
                        html.P(r["snippet"], style={"fontStyle": "italic"})
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            html.Pre(r["full_text"], style={"whiteSpace": "pre-wrap",
                                                            "maxHeight": "300px",
                                                            "overflowY": "auto"})
                        ),
                        id={"type": "collapse", "index": idx},
                        is_open=False
                    ),
                ],
                className="mb-3"
            )
        )
    return cards

@app.callback(
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    Input({"type": "collapse-btn", "index": MATCH}, "n_clicks"),
    State({"type": "collapse", "index": MATCH}, "is_open")
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run_server(host="0.0.0.0", port=port, debug=False)