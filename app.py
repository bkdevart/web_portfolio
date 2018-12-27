import os
from datetime import datetime, date
from math import pi
from flask import Flask, render_template
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.palettes import RdBu
from bokeh.transform import cumsum
import pandas as pd

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
    # grab source data
    source, complete = init_data()

    # add weekly hours for most played game plot
    plot = game_of_the_week(source)
    plots.append(components(plot))

    # add weekly hours distribution pie graph
    plot = weekly_hours_snapshot(source)
    plots.append(components(plot))

    return render_template('gameplay.html', plots=plots)


def init_data():
    '''
    Initializes data, returning two dataframes - source and complete
    '''
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    game_log = os.path.join(THIS_FOLDER, 'input/game_log.csv')
    completed = os.path.join(THIS_FOLDER, 'input/completed.csv')
    source = pd.read_csv(game_log,
                         parse_dates=['date', 'time_played'])
    complete = pd.read_csv(completed,
                           parse_dates=['date_completed'])

    # perform initial calculations
    source['minutes_played'] = ((source['time_played'].dt.hour * 60)
                                + source['time_played'].dt.minute)
    source['hours_played'] = source['minutes_played'] / 60
    source['dow'] = source['date'].dt.weekday_name
    source.sort_values('date', inplace=True)
    source['cum_total_minutes'] = (source.groupby('title')
                                   ['minutes_played'].cumsum())
    # add in start week (Sunday - Sat)
    # subtract the day number from the date (with 0 being Monday)
    source['week_start'] = (source['date'] - pd.to_timedelta
                            (source['date'].dt.dayofweek, unit='d'))
    source['month'] = source['date'].values.astype('datetime64[M]')

    return source, complete


def game_of_the_week(source_data, num_weeks=16):
    '''
    Shows each week's longest-played game and the hours spent playing for
    that week

    Parameters
    ----------
    num_weeks : int
        This determines the number of weeks to display, beginning from the
        current week and counting backwards.  Defaults to 16.

    Returns
    -------
    weekly_top_games : DataFrame
        The contents DataFrame are displayed graphically.  Fields below.

        week_start : datetime64[ns]
            week-start date (Monday-based) for top game of that week
        hours_played : float64
            Total hours played per game, per week
        title : object (str)
            Name of game played
    '''
    weekly_game_hours = (source_data.groupby(['week_start', 'title'],
                                             as_index=False)
                         [['hours_played']].sum())
    weekly_top_hours = (weekly_game_hours.groupby(['week_start'],
                                                  as_index=False).max()
                        [['week_start', 'hours_played']])
    # TODO: see if there's a more efficient way of doing this
    weekly_top_games = (weekly_top_hours
                        .merge(weekly_game_hours
                               [['week_start', 'hours_played', 'title']],
                               on=['week_start', 'hours_played'],
                               how='left'))

    graph = weekly_top_games[['title', 'hours_played']].tail(num_weeks)
    # plot graph df with bokeh
    source = ColumnDataSource(graph)
    y_range = list(set(graph['title']))

    plot = figure(plot_height=300,
                  sizing_mode='scale_width',
                  y_range=y_range,
                  title='Hours Per Week')
    plot.hbar(y='title',
              source=source,
              right='hours_played',
              height=.5,
              line_color='#8e8d7d',
              fill_color='#8e8d7d')
    most_recent_week = weekly_top_games['week_start'].dt.date.iloc[-1]
    curr_top_game = weekly_top_games['title'].iloc[-1]
    # TODO: convert this line to a string, figure out how display in HTML
    # top_game = f'Top Game for week of {most_recent_week}: {curr_top_game}'
    return plot


def weekly_hours_snapshot(source):
    '''
    Displays pie chart of time played per game in curretn week
    '''
    df = __weekly_hours_by_game(source)
    # filter to current week
    current_week = df['week_start'].max()
    df = (df[df['week_start'] == current_week]
          [['title', 'hours_played']])
    # some calculations to orientate the pie wedges
    df['angle'] = (df['hours_played']/
                   df['hours_played'].sum() * 2*pi)
    df['color'] = RdBu[len(df['hours_played'])]
    source = ColumnDataSource(df)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title='Weekly Hours Distribution')
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='title',
            fill_color='color', source=source)
    return p


def __weekly_hours_by_game(source):
    '''
    Shows total hours played for each game for each week

    Returns
    -------
    weekly_game_hours : DataFrame
        week_start : datetime64[ns]
            Week-start date (Monday-based) for title played
        title : object (str)
            Title of game
        hours_played : float64[ns]
            Total hours spent playing title for the week
    '''
    weekly_game_hours = (source.groupby(['week_start',
                                         'title'],
                                        as_index=False)
                         [['hours_played']].sum().sort_values(
                                                  ['week_start',
                                                   'hours_played'],
                                                  ascending=False))
    return weekly_game_hours


@app.route('/music/')
def music():
    return render_template('music.html')


@app.route('/exercise/')
def exercise():
    return render_template('exercise.html')
