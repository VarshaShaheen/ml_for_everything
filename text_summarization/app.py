from flask import Flask, request, render_template
from text_summarization import generate_summary  # Make sure this is accessible

app=Flask(__name__)

# routes
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/summarize_text', methods=['POST', 'GET'])
def summ_text():
    if request.method == 'POST':
        input_text = request.form['text']
        print(input_text)
        summary = generate_summary(input_text)
        return render_template('index.html', corrected_text=summary)
    return render_template('index.html')


@app.route('/summarize_file', methods=['POST','GET'])
def summ_file():
    pass

@app.route('/summarize_audio_file', methods=['POST','GET'])
def summ_audio():
    pass


if __name__ == '__main__':
    app.run(debug=True)