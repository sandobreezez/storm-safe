# Importing necessary libraries
import dash
from dash import html, dcc, Input, Output

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Input(id='txt_input', type='text', placeholder='Enter text here'),
    dcc.Input(id='num_input', type='number', placeholder='Enter a number', value=0),
    html.Div(id='txt_output'),
    html.Div(id='num_output')
])

# Define the callback for text input
@app.callback(
    Output('txt_output', 'children'),
    Input('txt_input', 'value')
)
def update_text_output(value):
    return f"You entered: {value}"

# Define the callback for numeric input
@app.callback(
    Output('num_output', 'children'),
    Input('num_input', 'value')
)
def update_num_output(value):
    if value is not None:
        return f"You entered the number: {round(value)}. Its square is: {round(value ** 2)}"
    return "Please enter a number"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
