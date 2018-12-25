from flask import Flask, render_template
from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gameplay/')
def gameplay():
    plots = []
    plots.append(make_plot())
    return render_template('gameplay.html', plots=plots)

def make_plot():
    plot = figure(plot_height=300, sizing_mode='scale_width')

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2**v for v in x]

    plot.line(x, y, line_width=4)

    script, div = components(plot)
    return script, div


@app.route('/music/')
def music():
    return render_template('music.html')


@app.route('/exercise/')
def exercise():
    return render_template('exercise.html')
