from flask import Flask, render_template, request, flash
from analysis import load_data, generate_plots, get_summary_stats, get_feature_columns, predict_quality
import os

app = Flask(__name__)
app.secret_key = 'wine-quality-secret-key'

# Generate plots once on startup (lazy — skip if already exist)
plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
required_plots = ['quality_dist.png', 'corr_red.png', 'corr_white.png',
                  'alcohol_quality.png', 'feature_hist.png']

if not all(os.path.exists(os.path.join(plots_dir, p)) for p in required_plots):
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
    errors = {}

    if request.method == 'POST':
        wine_type = request.form.get('wine_type', 'red')
        if wine_type not in ('red', 'white'):
            wine_type = 'red'

        # --- Validate each field ---
        FIELD_RANGES = {
            'fixed acidity':        (0.0, 20.0),
            'volatile acidity':     (0.0, 2.0),
            'citric acid':          (0.0, 2.0),
            'residual sugar':       (0.0, 70.0),
            'chlorides':            (0.0, 1.0),
            'free sulfur dioxide':  (0.0, 300.0),
            'total sulfur dioxide': (0.0, 500.0),
            'density':              (0.98, 1.04),
            'pH':                   (2.5, 4.5),
            'sulphates':            (0.0, 2.5),
            'alcohol':              (7.0, 18.0),
        }

        parsed_values = {}
        for feat in features:
            raw = request.form.get(feat, '').strip()
            input_values[feat] = raw

            if raw == '':
                errors[feat] = 'This field is required.'
                continue

            try:
                val = float(raw)
            except ValueError:
                errors[feat] = 'Must be a number.'
                continue

            lo, hi = FIELD_RANGES.get(feat, (None, None))
            if lo is not None and not (lo <= val <= hi):
                errors[feat] = f'Expected between {lo} and {hi}.'
                continue

            parsed_values[feat] = val

        if not errors:
            try:
                result = predict_quality(parsed_values, wine_type)
            except Exception as e:
                flash(f'Something went wrong during analysis: {str(e)}', 'error')
        else:
            flash('Please fix the errors below before submitting.', 'error')

    return render_template('predict.html',
                           features=features,
                           result=result,
                           wine_type=wine_type,
                           input_values=input_values,
                           errors=errors)


if __name__ == '__main__':
    app.run(debug=True)