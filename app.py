from flask import Flask, render_template

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


@app.route('/gameplay.html')
def gameplay():
    return render_template('gameplay.html')


@app.route('/music.html')
def music():
    return render_template('music.html')


@app.route('/exercise.html')
def exercise():
    return render_template('exercise.html')
