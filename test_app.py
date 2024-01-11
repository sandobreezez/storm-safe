from shiny import App, ui, render

# Define the app layout
app_ui = ui.page_fluid(
    ui.input_text("txt_input", "Enter text here:"),
    ui.input_numeric("num_input", "Enter a number:",value=0),
    ui.output_text("txt_output"),
    ui.output_text("num_output")
)

# Define the server logic
def server(input, output, session):
    @output
    @render.text
    def txt_output():
        return f"You entered: {input.txt_input()}"
    @render.text
    def num_output():
        # Accessing the number from the input
        entered_number = input.num_input()
        return f"You entered the number: {str(round(entered_number))}. \nIts square is: {str(round(entered_number ** 2))}"

# Create the app
app = App(app_ui, server)

# Run the app
if __name__ == "__main__":
    app.run()
