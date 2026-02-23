from flask import Flask, render_template, request
from analysis import load_data, generate_plots, get_summary_stats, get_feature_columns, predict_quality
import os

app = Flask(__name__)

# Generate plots once on startup
generate_plots()


@app.route('/')
def index():
    red, white, combined = load_data()
    red_stats = {
        'count': len(red),
        'avg_quality': round(red['quality'].mean(), 2),
        'avg_alcohol': round(red['alcohol'].mean(), 2),
    }
    white_stats = {
        'count': len(white),
        'avg_quality': round(white['quality'].mean(), 2),
        'avg_alcohol': round(white['alcohol'].mean(), 2),
    }
    return render_template('index.html', red=red_stats, white=white_stats)


@app.route('/analysis')
def analysis():
    red, white, _ = load_data()
    red_summary = get_summary_stats(red.drop(columns=['type']))
    white_summary = get_summary_stats(white.drop(columns=['type']))
    return render_template('analysis.html',
                           red_summary=red_summary,
                           white_summary=white_summary)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    features = get_feature_columns()
    wine_type = 'red'
    input_values = {f: '' for f in features}

    if request.method == 'POST':
        wine_type = request.form.get('wine_type', 'red')
        try:
            input_values = {f: request.form.get(f, '0') for f in features}
            result = predict_quality(input_values, wine_type)
        except Exception as e:
            result = {'error': str(e)}

    return render_template('predict.html',
                           features=features,
                           result=result,
                           wine_type=wine_type,
                           input_values=input_values)


if __name__ == '__main__':
    app.run(debug=True)